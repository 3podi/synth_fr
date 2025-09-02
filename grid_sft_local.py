import logging
import re
import subprocess
import uuid
import hydra
from omegaconf import DictConfig
import os
import sys

log = logging.getLogger(__name__)

def format_adapters(adapters: list[str]) -> str:
    return f"[{','.join(adapters)}]"

def to_str(adapters_path: list[str]) -> str:
    return f"[{','.join(adapters_path)}]"

def format_hydra_args(dataset: str, dataset_size: int, group_id: str, adapters: list[str], run_id: str, model_name: str) -> str:
    """
    Build a properly escaped Hydra override string for dataset, adapters, group ID, run ID, and model name.
    Important: dataset path values containing '=' must be escaped or quoted appropriately.
    """
    escape_dataset = dataset.replace("=", "\\=")
    return (
        f"\"group_id={group_id} run_id={str(run_id)} dataset_size={dataset_size} "
        f"'adapters_paths={to_str(adapters)}' "
        f"'dataset={escape_dataset}' "
        f"model_config.model_name_or_path={model_name}\""
    )

class JobManager:
    """Handles sequential execution of scripts."""
    def __init__(self):
        self.last_job_id = -1
    
    def submit(self, cmd: str) -> int:
        """Execute command sequentially."""
        return self.run_cmd(cmd)
    
    def run_cmd(self, cmd: str) -> int:
        log.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(result.stderr)
            raise RuntimeError(result.stderr)
        log.info(f"Command completed with return code: {result.returncode}")
        return result.returncode

class Pipeline:
    """Orchestrates the full pipeline by running scripts sequentially."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_steps = cfg.train_steps.split(" ")
        self.adapters = [] 
        self.group_id = cfg.group_id
        self.run_id = str(uuid.uuid4())[:7]
        self.job_mgr = JobManager()
        self.sts_model_path = self.cfg.sts_model.replace("/", "-")
        self.model_name = cfg.model_name.replace("/", "-")
    
    def base_model_path(self) -> str:
        return (
            f"datasets/{self.cfg.domain}/"
            f"model={self.model_name}_size={self.cfg.size}"
        )
    
    def output_model_path(self) -> str:
        return (
            f"datasets/{self.cfg.domain}/"
            f"model={self.run_id}_size={self.cfg.size}_step={self.train_steps[-1]}"
        )
    
    def dataset_path(self) -> str:
        return (
            f"{self.base_model_path()}/public_seed.parquet"
        )

    def run_train_steps(self):
        """Run training step."""
        script = self.cfg.scripts[self.cfg.step]
        dataset = self.dataset_path()

        # Create hydra arguments as a proper string
        hydra_args = (
            f"group_id={self.group_id} "
            f"run_id={self.run_id} "
            f"dataset_size={self.cfg.size_sft} "
            f"adapters_paths={to_str(self.adapters)} "
            f"dataset={dataset.replace('=', '\\=')} "
            f"model_config.model_name_or_path={self.cfg.model_name}"
        )

        cmd = f"python3 {script} --config-name {self.cfg.domain} {hydra_args}"

        log.info(f"Executing training step: {cmd}")
        self.job_mgr.submit(cmd)
        
        # Update adapters list for next steps
        self.adapters.append(f"lora/{self.cfg.step}/{run_id}")
    
    def run_generation(self):
        """Run generation step."""

        cmd1 = (
        
            f"python3 {self.cfg.scripts.merge} "
            f"--model {self.cfg.model_name} "
            f"--adapters {','.join(self.adapters)} "
            f"--output_path ./lora/merge/{self.run_id} "
        ) 
        log.info(f"Executing merging step: {cmd1}")
        self.job_mgr.submit(cmd1)
        
        cmd2 = (
            f"python3 {self.cfg.scripts.generation} "
            f"--dataset {self.cfg.private_path} "
            f"--model ./lora/merge/{self.run_id} "
            f"--output_path {self.base_model_path()} "
            f"--num_prompts {self.cfg.size_generation} "
            f"--tp {self.cfg.tp} "
            f"--pp {self.cfg.pp} "
        ) 
        log.info(f"Executing generation step: {cmd2}")
        self.job_mgr.submit(cmd2)
        
        cmd3 = (
            f"python3 {self.cfg.scripts.random_generation} "
            f"--model ./lora/merge/{self.run_id} "
            f"--output_path {self.base_model_path()} "
            f"--csv_path {self.cfg.csv_path} "
            f"--max_codes {self.cfg.max_codes} "
            f"--max_kws {self.cfg.max_kws} "
            f"--num_samples {self.cfg.num_samples_rand_gen} "
            f"--tp {self.cfg.tp} "
            f"--pp {self.cfg.pp} "
        ) 
        log.info(f"Executing random generation step: {cmd3}")
        self.job_mgr.submit(cmd3)
    
    def run_score(self):
        """Run scoring step."""        
        cmd = (
            f"python3 {self.cfg.scripts.score} "
            f"--sts_model {self.cfg.sts_model} "
            f"--public_dataset {self.base_model_path()} "
            f"--private_dataset {self.cfg.private_path} "
            f"--output_path {self.base_model_path()} "
            f"--tp {selg.cfg.tp} "
            f"--n 4 "
            f"--wdb_id {self.run_id} "
            f"--group_id {self.group_id} "
            f"--model_name {self.cfg.model_name} "
        )
        
        log.info(f"Executing scoring step: {cmd}")
        self.job_mgr.submit(cmd)
    
    def run_filter(self):
        """Run filtering step."""
        path = f"{self.base_model_path()}/model={self.sts_model_path}_scored.parquet"
        cmd = f"python3 {self.cfg.scripts.filter} input_file {path}"
        
        log.info(f"Executing filtering step: {cmd}")
        self.job_mgr.submit(cmd)
    
    def run_classification(self):
        """Run classification step."""
        script = self.cfg.scripts.classification
        dataset = f"{self.base_model_path()}/random_dataset.parquet"
        run_id = self.run_id
        
        hydra_args = (
            f"dataset={dataset.replace('=', '\\=')} "
        )
        
        cmd = f"python3 {script} {hydra_args}"
        
        log.info(f"Executing classification step: {cmd}")
        self.job_mgr.submit(cmd)
        
@hydra.main(config_path=".", config_name="grid_sft.yaml", version_base="1.3")
def main(cfg: DictConfig):
    pipeline = Pipeline(cfg)
    pipeline.run_train_steps()
    pipeline.run_generation()
    pipeline.run_score()
    pipeline.run_filter()
    pipeline.run_classification()


if __name__ == "__main__":
    main()
