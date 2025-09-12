import logging
import re
import subprocess
import uuid
import os
import hydra
from omegaconf import DictConfig
log = logging.getLogger(__name__)

def get_adapters_list(lora_path: str, run_id: str, start_iter: int):
    adapters = []
    adapters.append(os.path.join(lora_path, 'sft', run_id))
    if start_iter > 1:
        dpo_adapters = os.listdir(os.path.join(lora_path, 'dpo', run_id))
        print('Found adapters from previous dpo iterations: ', dpo_adapters)
        for adpt in dpo_adapters:
            adapters.append(os.path.join(lora_path, 'dpo', run_id, adpt))
    return adapters

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
        
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Log each line as it comes in
                log.info(output.strip())
        
        # Get the return code
        return_code = process.poll()
        
        if return_code != 0:
            log.error(f"Command failed with return code: {return_code}")
            raise RuntimeError(f"Command failed: {cmd}")
        
        log.info(f"Command completed with return code: {return_code}")
        return return_code

class Pipeline:
    """Orchestrates the full pipeline by running scripts sequentially."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_steps = cfg.train_steps.split(" ")
        self.adapters = get_adapters_list(
            lora_path=cfg.lora_path,
            run_id=cfg.run_id,
            start_iter=cfg.start_it
        )
        self.group_id = cfg.group_id
        self.run_id = cfg.run_id
        self.job_mgr = JobManager()
        self.sts_model_path = self.cfg.sts_model.replace("/", "-")
        self.model_name = cfg.model_name.replace("/", "-")
    
    def base_model_path(self) -> str:
        return (
            f"datasets/{self.cfg.domain}/"
            f"model={self.model_name}_size={self.cfg.size}"
        )
    
    def output_file_path(self, iter) -> str:
        return f"{self.base_model_path()}/dpo_iter_{iter}"
    
    def dataset_path(self, iter) -> str:
        if iter == 1:
            return (
                f"{self.base_model_path()}/model={self.sts_model_path}_scored_dpo_{self.cfg.sorting}.parquet"
            )
        else:
            return f"{self.base_model_path()}/dpo_iter_{iter-1}/model={self.sts_model_path}_scored_dpo_{self.cfg.sorting}.parquet"
    
    def run_train_steps(self, iter):
        script = self.cfg.scripts[self.cfg.step]
        dataset = self.dataset_path(iter)

        escaped_dataset = dataset.replace("=", "\\=")
        # Create hydra arguments as a proper string
        hydra_args = (
            f"group_id={self.group_id} "
            f"run_id={self.run_id} "
            f"iteration={iter} "
            f"dataset_size={self.cfg.size} "
            f"'adapters_paths={to_str(self.adapters)}' "
            f"'dataset={escaped_dataset}' "
            f"'model_config.model_name_or_path={self.cfg.model_name}'"
        )

        cmd = f"python3 {script} --config-name {self.cfg.domain} {hydra_args}"
        self.job_mgr.submit(cmd)
        self.adapters.append(f"lora/{self.cfg.step}/{self.run_id}/iteration_{iter}")
    
    def run_generation(self, iter):
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
            f"--output_path {self.output_file_path(iter)} "
            f"--num_prompts {self.cfg.size_generation} "
            f"--tp {self.cfg.tp} "
            f"--pp {self.cfg.pp} "
        ) 
        log.info(f"Executing generation step: {cmd2}")
        self.job_mgr.submit(cmd2)
        
        cmd3 = (
            f"python3 {self.cfg.scripts.random_generation} "
            f"--model ./lora/merge/{self.run_id} "
            f"--output_path {self.output_file_path(iter)} "
            f"--csv_path {self.cfg.csv_path} "
            f"--max_codes {self.cfg.max_codes} "
            f"--max_kws {self.cfg.max_kws} "
            f"--num_samples {self.cfg.num_samples_rand_gen} "
            f"--tp {self.cfg.tp} "
            f"--pp {self.cfg.pp} "
        ) 
        log.info(f"Executing random generation step: {cmd3}")
        self.job_mgr.submit(cmd3)
    
    def run_score(self, iter):
        """Run scoring step."""
        cmd = (
            f"python3 {self.cfg.scripts.score} "
            f"--sts_model {self.cfg.sts_model} "                             
            f"--public_dataset {self.output_file_path(iter)} "
            f"--output_path {self.output_file_path(iter)} "
            f"--private_dataset {self.cfg.private_path} "
            f"--n 4 "
            f"--wdb_id {self.run_id} "
            f"--group_id {self.group_id} "
            f"--model_name {self.cfg.model_name} "
        )
        self.job_mgr.submit(cmd)
    
    def run_filter(self, iter):
        path = f"{self.output_file_path(iter)}/model={self.sts_model_path}_scored.parquet"
        self.job_mgr.submit(f"python3 {self.cfg.scripts.filter} {path}")
    
    def run_eval(self):
        step = self.train_steps[-1]
        path = f"{self.output_model_path()}/model={self.sts_model_path}_scored_eval.parquet"
        out_path = (
            f"datasets/health/eval/model_outputs/"
            f"model={self.run_id}_size={self.cfg.size}_step={step}"
        )
        eval_gen_cmd = (
            f"python3 {self.cfg.scripts.eval_gen} --DOWNSTREAM_DS_PATH {path} "
            f"--OUTPUT_PATH {out_path} --GROUP_ID {self.group_id}"
        )
        self.job_mgr.submit(eval_gen_cmd)
        eval_pref_cmd = (
            f"python3 {self.cfg.scripts.eval_preference} --MODEL_ID {self.run_id} --STEP {step} "
            f"--SIZE {self.cfg.size} "
            f"--SUFFIX_RUN_NAME {'-'.join(self.train_steps)}-{self.cfg.sorting} "
            f"--GROUP_ID {self.group_id}"
        )
        self.job_mgr.submit(eval_pref_cmd)

@hydra.main(config_path=".", config_name="grid_rl.yaml", version_base="1.3")
def main(cfg: DictConfig):
    pipeline = Pipeline(cfg)
    for iter in range(cfg.start_it, cfg.end_it):
        lora_dpo_path = f'lora/{cfg.step}/{cfg.run_id}/iteration_{iter}'
        if os.path.isdir(lora_dpo_path):
            print('Already found adapter for iter ', iter)
            print('Skipping dpo iter ', iter)
            pipeline.adapters.append(f"lora/{cfg.step}/{cfg.run_id}/iteration_{iter}")
        else:
            pipeline.run_train_steps(iter)
            pipeline.run_generation(iter)
            pipeline.run_score(iter)
            pipeline.run_filter(iter)
    #pipeline.run_eval()

if __name__ == "__main__":
    main()