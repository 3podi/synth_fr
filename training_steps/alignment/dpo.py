import hydra
import peft
import torch
import wandb
import pandas as pd
from datasets import Dataset
from omegaconf import ListConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

import sys
from transformers import TrainerCallback
class PrintAllLogsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs, **kwargs):
        print(f"[Step {state.global_step}] Logs: {logs}")
        sys.stdout.flush()

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg):
    """Train the model using the reinforcement learning algorithm DPO.
    We fix the percentile of the best candidate to keep for training.

    Args:
        cfg: The configuration for the training.
    """

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
        group=f"{cfg.group_id}",
    )
    
    print("---DPO starting---")
    print("Iter: ", cfg.iteration)
    print("Adapters: ", cfg.adapters_paths)
    model_config = hydra.utils.instantiate(cfg.model_config)
    dpo_config = hydra.utils.instantiate(cfg.dpo_config)
    peft_config = hydra.utils.instantiate(cfg.peft_config)
    dpo_config.group_by_length = False

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    peft_config.target_modules = (
        list(peft_config.target_modules)
        if isinstance(peft_config.target_modules, ListConfig)
        else peft_config.target_modules
    )
    model_kwargs = dict(
        torch_dtype=torch_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    merge_adapters(model, cfg.adapters_paths)
    model = peft.get_peft_model(
        model,
        peft_config,
    )

    model.add_adapter(peft_config=peft_config, adapter_name="reference")
    model.enable_input_require_grads()
    
    print(model)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]    
    num_params = sum(p.numel() for p in params_to_optimize)
    num_params_mb = sum(p.numel() * p.element_size() for p in params_to_optimize) / (1024**2)
    print(f"Trainable parameters: {num_params:,}")
    print(f"Total size: {num_params_mb:.2f} MB")

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    dpo_config.padding_value = tokenizer.eos_token_id

    dataset = Dataset.from_pandas(pd.read_parquet(cfg.dataset).head(cfg.dataset_size))
    # Add prompt column
    dataset = dataset.add_column("prompt", dataset["instruction"])
    dataset = dataset.select_columns(["prompt", "chosen", "rejected"])
    
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    print('len train: ', len(train_dataset))
    print('len val: ', len(val_dataset))

    dpo_trainer = DPOTrainer(
        args=dpo_config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[PrintAllLogsCallback()],
    )
    dpo_trainer.train()
    #dpo_path = f"lora/dpo-{cfg.iteration}/{wandb.run.id}"
    dpo_path = f"lora/dpo/{cfg.run_id}/iteration_{cfg.iteration}"
    dpo_trainer.save_model(dpo_path)


def merge_adapters(model, adapter_paths):
    for adapter_path in adapter_paths:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    return model


if __name__ == "__main__":
    main()
