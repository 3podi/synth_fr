from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from trl import GRPOConfig, GRPOTrainer
import hydra
from utils_grpo import format_grpo2, reward_match_format_exactly, reward_f1, reward_no_repeat, reward_matching_keywords2, ScoringModelRewardFunction, ScoreModelAPI
from omegaconf import DictConfig, ListConfig, OmegaConf
import wandb
from datasets import Dataset

from sentence_transformers import SentenceTransformer


@hydra.main(version_base=None, config_path="./configs", config_name="grpo")
def main(cfg):
    """Train the model using the reinforcement learning algorithm GRPO.

    Args:
        cfg: The configuration for the training.
    """
    
    wandb_config = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )
    wandb.init(
        project="synth-kg-grpo",
        tags=cfg.tags,
        config=wandb_config,
        job_type="training",
        group=cfg.group_id,
        settings=wandb.Settings(init_timeout=100)
    )
    
    max_seq_length = 2048 # Can increase for longer reasoning traces
    lora_rank = 64 # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        #model_name = "Qwen/Qwen2.5-3B-Instruct",
        model_name = cfg.model_name,
        max_seq_length = max_seq_length, #shouldnt be important if i aim at generating shorter reports
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.7, # Reduce if out of memory
    )
    
    if cfg.adapter_path is None or len(cfg.adapter_path) == 0:
    
        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ], # Remove QKVO if out of memory
            lora_alpha = lora_rank,
            use_gradient_checkpointing = "unsloth", # Enable long context finetuning
            random_state = 3407,
        )
    else:
        
        if isinstance(cfg.adapter_path, str):
            cfg.adapter_path = [cfg.adapter_path]
        
        for path in cfg.adapter_path:
            model.load_adapter(path)
    
    training_args = GRPOConfig(
        use_vllm = True, # use vLLM for fast inference!
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Increase to 4 for smoother training
        num_generations = 6, # Decrease if out of memory
        max_prompt_length = 256, #### prompt gets truncated from left
        max_completion_length = 2048, ### max length of the completion, should set based on number of tokens in reports (or the length of reports i want)
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 250,
        save_steps = 250,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = "outputs",
    )

    #### DATA ####
    #can make a parquet from extraction results file, only need a map code2integer_id
    dataset = Dataset.from_parquet(cfg.dataset)
    max_size = min(cfg.dataset_size, len(dataset))
    dataset = dataset.select(range(max_size))
    dataset = dataset.map(format_grpo2)
    
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    del dataset
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    del split_dataset
    print('Len train dataset: ', len(train_dataset))
    print('Len val dataset: ', len(val_dataset))
    
    #### REWARD ####
    
    #sts_model = SentenceTransformer("FremyCompany/BioLORD-2023-M")
    educational_score_reward = ScoringModelAPI(sts_model=cfg.sts_model)
    
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            #reward_match_format_exactly, 
            #reward_f1, 
            #reward_no_repeat, 
            educational_score_reward,
            reward_matching_keywords2
        ],
        args = training_args,
        train_dataset = train_dataset,
    )
    trainer.train()
    
if __name__ == "__main__":
    main()
