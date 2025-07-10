import logging

import hydra
import pandas as pd
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, get_peft_config

from transformers.data.data_collator import DefaultDataCollator

import os
import sys

logger = logging.getLogger(__name__)

from transformers import TrainerCallback
class PrintAllLogsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs, **kwargs):
        print(f"[Step {state.global_step}] Logs: {logs}")
        sys.stdout.flush()


def chunk_tokens_with_padding(example, padding_token, chunk_size=512, stride=448):
    input_ids = example["input_ids"]
    chunks = []

    for i in range(0, len(input_ids), stride):
        chunk = input_ids[i : i + chunk_size]
        chunk_len = len(chunk)
        
        # Pad if needed
        if chunk_len < chunk_size:
            pad_length = chunk_size - chunk_len
            chunk += [padding_token] * pad_length
            #print('Len padded chunk: ', len(chunk))
            #print('N token in un padded chunk: ', chunk_len)
            chunk_len += 1    #include eos token in the attention mask
            #print('N tokens for loss: ', chunk_len)
            #print('labels: ',chunk[:chunk_len])
        
        labels = chunk[:chunk_len] + [-100] * (chunk_size - chunk_len) #-100 to not compute loss on pad tokens
        
        chunks.append({
            "input_ids": chunk,
            "attention_mask": [1] * chunk_len + [0] * (chunk_size - chunk_len), 
            "labels": labels,
        })
    return {"chunk": chunks}
    #return chunks
    
@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig):
    """Train the model with supervised fine-tuning.

    This function loads a pre-trained model, applies supervised fine-tuning using the provided
    dataset, and evaluates the model on a test dataset.

    Args:
        cfg (DictConfig): The configuration for the training, containing hyperparameters
        and settings.

    Note:
        This function uses the SFTTrainer from the TRL library for supervised fine-tuning.
        It also integrates with Weights & Biases (wandb) for experiment tracking.
    """
    
    custom_run_id = cfg.run_id
    os.environ["WANDB_RUN_ID"] = custom_run_id
    
    wandb_config = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )
    wandb.init(
        project="synth-kg",
        tags=cfg.tags,
        config=wandb_config,
        job_type="training",
        group=cfg.group_id,
        settings=wandb.Settings(init_timeout=100),
    )
    print('Wandb run-id: ', wandb.run.id)
    print("SFT model: ", cfg.model_config.model_name_or_path)
    model_config = hydra.utils.instantiate(cfg.model_config)
    sft_config = hydra.utils.instantiate(cfg.sft_config)
    cfg.sft_config.output_dir = f"lora/sft/{cfg.run_id}"
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        use_cache=False #if sft_config.gradient_checkpointing else True, never using cache at training time
    )
    model_config.lora_target_modules = (
        list(model_config.lora_target_modules)
        if isinstance(model_config.lora_target_modules, ListConfig)
        else model_config.lora_target_modules
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'Using padding token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}')
    print(f'Using eos token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}')
        
    def format_and_tokenize(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=cfg.sft_config.max_seq_length,
        )

    def format_and_tokenize2(example):
        return tokenizer(
            example["text"],
            #padding="max_length",
            truncation=False,
            #max_length=cfg.sft_config.max_seq_length,
        )
    
    
    #params_to_optimize = [p for p in model.parameters() if p.requires_grad]    
    #num_params = sum(p.numel() for p in params_to_optimize)
    #num_params_mb = sum(p.numel() * p.element_size() for p in params_to_optimize) / (1024**2)
    #print(f"Trainable parameters: {num_params:,}")
    #print(f"Total size: {num_params_mb:.2f} MB")

    dataset = Dataset.from_pandas(pd.read_parquet(cfg.dataset).head(cfg.dataset_size))
    
    #def rename_columns(example):
    #    return {
    #        "prompt": example["instruction"],
    #        "completion": example["response"],
    #    }
    #dataset = dataset.map(rename_columns, remove_columns=dataset.column_names)
    
    dataset = dataset.map(lambda x: {"text": x["instruction"] + x["response"]})
    dataset = dataset.map(format_and_tokenize2, batched=False)
    
    dataset = dataset.map(
        lambda x: chunk_tokens_with_padding(x, tokenizer.pad_token_id),
        remove_columns=dataset.column_names,
        batched=False,
    ).flatten()  
    
    #dataset = dataset.map(
    #    lambda x: x['chunk'],
    #    remove_columns=dataset.column_names,
    #    batched=True,
    #)
    #dataset = dataset['chunk']
    print('Len dataset: ', len(dataset))
    
    #dataset = dataset.map(lambda x: {"n_tokens": len(x['input_ids'])})    
    #print(dataset['n_tokens'])
    
    #sample_index = 0
    #input_ids = dataset[sample_index]['input_ids']
    # Decode the input_ids back to text (skip special tokens like padding if needed)
    #decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    #print(decoded_text)
    
    #dataset = StreamingChunkedDataset(cfg.dataset,cfg.dataset_size,tokenizer)

    
    def chunks_generator(original_dataset):
        for example in original_dataset:
            for chunk_dict in example['chunk']:
                yield chunk_dict

    dataset = Dataset.from_generator(lambda: chunks_generator(dataset))
    #column_names = list(next(iter(dataset)).keys())
    #print('aaa: ', column_names)
    
    #dataset = dataset.map(lambda x: {"n_tokens": len(x['input_ids'])})
    #print(dataset["n_tokens"])
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    print('len train: ', len(train_dataset))
    print('len val: ', len(val_dataset))
    
    data_collator = DefaultDataCollator(return_tensors="pt")
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
        callbacks=[PrintAllLogsCallback()],
        data_collator=data_collator,
    )
    
    #train_dataloader = trainer.get_train_dataloader()
    #first_batch = next(iter(train_dataloader))
    #print('First batch:', first_batch)
    
    #print(first_batch['input_ids'][0])
    #print(first_batch['labels'][0])
    #print(len(first_batch['input_ids'][0]))
    #print(len(first_batch['labels'][0]))    
    
    #input_ids_batch = first_batch["input_ids"] 
    #decoded_texts = [tokenizer.decode(input_ids, skip_special_tokens=False) for input_ids in input_ids_batch]
    #print(decoded_texts)
    
    #column_names = list(next(iter(train_dataloader)).keys())
    #print('names: ', column_names)
    #print('is processed: ', 'input_ids' in column_names)

    trainer.train()
    trainer.save_model(cfg.sft_config.output_dir)

if __name__ == "__main__":
    main()
