import logging
import hydra
import pandas as pd
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers.data.data_collator import DefaultDataCollator
import os
import sys
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from typing import List, Dict, Any
import json
from collections import Counter

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    roc_auc_score,
    jacc
    )
logger = logging.getLogger(__name__)


# --- Compute pos_weight for each label based on TRAINING SET ---
def compute_pos_weights(train_dataset):
    """
    Compute positive weights for each label: num_neg / num_pos
    Avoid division by zero by setting a minimum positive count (e.g., 1)
    """
    # Stack all label tensors
    all_labels = np.stack([example['labels'] for example in train_dataset])

    # Count positive samples per label
    pos_counts = np.sum(all_labels, axis=0)  # shape: (num_labels,)
    total_samples = len(train_dataset)
    neg_counts = total_samples - pos_counts

    # Avoid division by zero if no positive samples, set weight to 1.0
    pos_counts = np.clip(pos_counts, 1, None)  # minimum 1 to avoid inf

    pos_weights = neg_counts / pos_counts
    return torch.tensor(pos_weights, dtype=torch.float32)


def compute_metrics(eval_pred: EvalPrediction):
    """
    Compute label-wise and sample-wise metrics for multi-label classification.
    Focuses on non-exact-match evaluation since labels are independent.

    Args:
        eval_pred: EvalPrediction with .predictions (logits) and .label_ids (multi-hot)

    Returns:
        dict: Comprehensive metrics for multi-label performance
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # Apply sigmoid to get probabilities (since labels are independent)
    sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -100, 100)))  # clip for stability
    probs = sigmoid(predictions)
    preds = (probs > 0.5).astype(int)  # threshold at 0.5

    labels = labels.astype(int)

    # --- Metrics ---
    h_loss = hamming_loss(labels, preds)
    jaccard_macro = jaccard_score(labels, preds, average="samples", zero_division=0)

    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)

    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    precision_micro = precision_score(labels, preds, average="micro", zero_division=0)
    recall_micro = recall_score(labels, preds, average="micro", zero_division=0)

    return {
        "hamming_loss": h_loss,
        "jaccard_score_samples_avg": jaccard_macro,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_micro": f1_micro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
    }

class PrintAllLogsCallback:
    def on_log(self, args, state, control, logs, **kwargs):
        print(f"[Step {state.global_step}] Logs: {logs}")
        sys.stdout.flush()

def count_labels_in_dataset(df: pd.DataFrame, label_column: str) -> Dict[str, int]:
    """
    Count occurrences of each label in the dataset.
    Assumes labels are whitespace-separated in the label_column.
    """
    label_counts = Counter()
    
    for _, row in df.iterrows():
        if pd.notna(row[label_column]):
            # Split by whitespace and strip any extra spaces
            labels = [label.strip()[:3] for label in str(row[label_column]).split() if label.strip()]
            label_counts.update(labels)
    
    return dict(label_counts)

def filter_labels_by_frequency(label_counts: Dict[str, int], percentage: float, n_keep: int) -> List[str]:
    """
    Filter labels by their occurrence percentage.
    """
    if not label_counts:
        return []
    
    # Sort labels by frequency (descending)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate number of labels to keep
    if percentage:
        print(f'Keeping first {percentage}% of labels')
        num_labels_to_keep = int(len(sorted_labels) * percentage / 100)
    else:
        print(f'Keeping first {n_keep} labels')
    
    # Keep the most frequent labels
    filtered_labels = [label for label, freq in sorted_labels[:num_labels_to_keep]]
    
    print(f"Kept {len(filtered_labels)} out of {len(label_counts)} labels") #({percentage}% most frequent)")
    print(f"Top 10 most frequent labels: {sorted_labels[:10]}")
    return filtered_labels

def prepare_multilabel_data(df: pd.DataFrame, text_column: str, label_column: str, 
                           label_list: List[str], max_labels_per_sample: int = 5) -> Dataset:
    """
    Prepare data for multi-label classification.
    Assumes labels are whitespace-separated strings in label_column.
    """
    # Create a list of samples
    samples = []
    
    for _, row in df.iterrows():
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        
        # Handle labels - split by whitespace and clean
        if pd.isna(row[label_column]):
            labels = []
        else:
            labels = [label.strip() for label in str(row[label_column]).split() if label.strip()]
            
        # Filter labels to only include those in our label list
        filtered_labels = [label for label in labels if label in label_list]
        
        # Limit number of labels per sample
        if len(filtered_labels) > max_labels_per_sample:
            filtered_labels = filtered_labels[:max_labels_per_sample]
            
        # Only keep samples that have at least one filtered label
        if len(filtered_labels) > 0:
            samples.append({
                "text": text,
                "labels": filtered_labels
            })
    
    return Dataset.from_list(samples)

def prepare_labels(batch, label_list: List[str]) -> Dict[str, Any]:
    """
    Prepare binary labels for multi-label classification.
    This function filters labels to only include those in the label_list,
    and creates a binary matrix for multi-label classification.
    """
    # Create MultiLabelBinarizer for the specific label set
    mlb = MultiLabelBinarizer(classes=label_list)
    
    # Transform labels to binary matrix
    binary_labels = mlb.fit_transform(batch["labels"])
    
    # Convert to tensor
    return {
        "labels": torch.tensor(binary_labels, dtype=torch.float32)
    }

def tokenize_function(examples, tokenizer, max_length: int = 512):
    """Tokenize examples for multi-label classification."""
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            # pos_weight expects 1D tensor of shape [num_labels]
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

class MultiEvalSFTTrainer(WeightedTrainer):
    def __init__(self, *args, eval_datasets: Dict[str, Any] = None, **kwargs):
        eval_datasets = eval_datasets or {}

        # Pick main dataset
        if "main" in eval_datasets:
            main_eval = eval_datasets["main"]
        elif len(eval_datasets) > 0:
            first_key = next(iter(eval_datasets))
            main_eval = eval_datasets[first_key]
            eval_datasets["main"] = main_eval
        else:
            main_eval = None

        # Ensure base Trainer sees something
        if main_eval is not None and "eval_dataset" not in kwargs:
            kwargs["eval_dataset"] = main_eval

        super().__init__(*args, **kwargs)

        # Save dict of all eval datasets
        self.eval_datasets = eval_datasets

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Run base evaluation on "main"
        main_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        all_metrics = dict(main_metrics)

        # Extra evals
        for name, dataset in self.eval_datasets.items():
            if name == "main":
                continue

            print(f"Evaluating on {name} dataset...")
            dataloader = self.get_eval_dataloader(dataset)

            eval_preds = self.evaluation_loop(
                dataloader,
                description=f"Evaluation {name}",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )

            metrics = {}
            if self.compute_metrics is not None and eval_preds.predictions is not None:
                eval_prediction = EvalPrediction(
                    predictions=eval_preds.predictions,
                    label_ids=eval_preds.label_ids,
                )
                metrics = self.compute_metrics(eval_prediction)

            # Always include loss from the loop
            metrics.update(eval_preds.metrics)

            # Prefix
            prefixed_metrics = {f"{name}_{k}": v for k, v in metrics.items()}
            all_metrics.update(prefixed_metrics)

            # Optional W&B logging
            if "wandb" in globals() and wandb.run is not None:
                wandb.log(prefixed_metrics, commit=False)

        if "wandb" in globals() and wandb.run is not None:
            wandb.log({}, commit=True)


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig):
    """
    Multi-label classification training script.
    This function trains a model to predict multiple labels for each input text.
    Labels are automatically extracted from the training dataset and filtered by frequency.
    """
    
    custom_run_id = cfg.run_id
    os.environ["WANDB_RUN_ID"] = custom_run_id
    wandb_config = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )
    
    wandb.init(
        project="synth-fr",
        tags=cfg.tags,
        config=wandb_config,
        job_type="training",
        group=cfg.group_id,
        settings=wandb.Settings(init_timeout=100),
    )
    
    print('Wandb run-id: ', wandb.run.id)
    print("Model: ", cfg.model_config.model_name_or_path)

    # Load and prepare dataset
    print(f"Loading dataset from {cfg.dataset}")
    df = pd.read_parquet(cfg.dataset)

    # Limit dataset size if specified
    if hasattr(cfg, 'dataset_size') and cfg.dataset_size > 0:
        df = df.head(cfg.dataset_size)

    print(f"Dataset size: {len(df)} samples")

    # Count labels in the dataset to get frequencies
    print("Counting label frequencies from dataset...")
    label_counts = count_labels_in_dataset(df, cfg.label_column)
    print(f"Found {len(label_counts)} unique labels in dataset")

    # Filter labels by frequency
    filtered_labels = filter_labels_by_frequency(
        label_counts,
        cfg.label_filter_percentage,
        cfg.label_filter_number
    )

    # If no labels were selected, show warning and exit
    if len(filtered_labels) == 0:
        print("Warning: No labels selected after filtering. Exiting.")
        return
    
    # Prepare multi-label data with filtering
    dataset = prepare_multilabel_data(
        df,
        cfg.text_column,
        cfg.label_column,
        filtered_labels,
        max_labels_per_sample=cfg.max_labels_per_sample
    )

    print(f"Prepared dataset with {len(dataset)} samples (after filtering)")

    # Show label distribution in final dataset
    all_labels_in_dataset = set()
    for sample in dataset:
        all_labels_in_dataset.update(sample['labels'])

    print(f"Labels present in final dataset: {len(all_labels_in_dataset)}")
    print(f"Sample labels: {sorted(list(all_labels_in_dataset))[:10]}...")  # Show first 10


    print(f"Loading SECOND validation dataset from {cfg.val_dataset_2}")
    df_val2 = pd.read_parquet(cfg.val_dataset_2)

    # Apply SAME label filtering
    df_val_2 = prepare_multilabel_data(
        df_val2,
        cfg.text_column,
        cfg.label_column,
        filtered_labels,  #reuse train's filtered labels
        max_labels_per_sample=cfg.max_labels_per_sample
    )

    print(f"Prepared SECOND validation dataset with {len(df_val_2)} samples (after filtering)")
    if len(df_val_2) == 0:
        print("Warning: Second validation dataset is empty after filtering.")
        df_val_2 = None

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'Using padding token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}')
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, cfg.max_seq_length),
        batched=True,
        remove_columns=["text"]  # Remove original columns
    )

    df_val_2 = df_val_2.map(
        lambda examples: tokenize_function(examples, tokenizer, cfg.max_seq_length),
        batched=True,
        remove_columns=["text"]  # Remove original columns
    )

    # Prepare labels for multi-label classification
    # Fit the binarizer once using all labels
    mlb = MultiLabelBinarizer(classes=filtered_labels)
    mlb.fit([filtered_labels])

    def add_binary_labels(example):
        binary_labels = mlb.transform([example["labels"]])[0]  # shape (num_labels,)
        return {"labels": binary_labels.astype("float32")}

    def add_binary_labels2(example):
        # Create binary vector for multi-label classification
        binary_labels = [1.0 if label in example["labels"] else 0.0 for label in filtered_labels]
        return {"labels": torch.tensor(binary_labels, dtype=torch.float32)}

    # Apply label preparation
    labeled_dataset = tokenized_dataset.map(
        add_binary_labels2,
        batched=False
    )

    df_val_2 = df_val_2.map(
        add_binary_labels2,
        batched=False
    )

    # Split dataset
    split_dataset = labeled_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']

    print('Num train samples: ', len(train_dataset))
    print('Num val samples: ', len(val_dataset))
    print('Num samples in val 2: ', len(df_val_2))
    
    # Initialize model for multi-label classification
    model_kwargs = dict(
        torch_dtype=cfg.model_config.torch_dtype,  
        num_labels=len(filtered_labels),  # Number of labels for classification
        problem_type="multi_label_classification",  # Specify multi-label classification
        pad_token_id=tokenizer.eos_token_id,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_config.model_name_or_path, 
        **model_kwargs
    )
    
    # Configure training arguments
    training_args = TrainingArguments(
        run_name=cfg.run_id,
        **cfg.training_arguments
    )
    
    # Create data collator
    data_collator = DefaultDataCollator(return_tensors="pt")

    class_weights = compute_pos_weights(train_dataset)
    # Initialize trainer
    trainer = MultiEvalSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_datasets={
        "main": val_dataset,
        "private_eval": df_val_2,
        },
        tokenizer=tokenizer,
        data_collator=data_collator,
        #callbacks=[PrintAllLogsCallback()],
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        #max_grad_norm=10.0
    )
    
    # Train the model
    trainer.train()
    
    if cfg.save_flag:
        # Save the model
        trainer.save_model(f"models/{cfg.run_id}")
    
        # Save label mappings
        label_mapping = {
            "filtered_labels": filtered_labels,
            "label_filter_percentage": cfg.label_filter_percentage
        }
    
        with open(f"models/{cfg.run_id}/label_mapping.json", "w") as f:
            json.dump(label_mapping, f, indent=2)
    
    print("Training completed!")
    wandb.finish()

if __name__ == "__main__":
    main()