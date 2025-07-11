import logging
import re
import subprocess
import uuid
import os

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def get_adapters_list(lora_path:str, run_id: str, start_iter: int):
    
    adapters = []
    adapters.append(os.path.join(lora_path,'sft',run_id))
    
    if start_iter > 1:
        dpo_adapters = os.listdir(os.path.join(lora_path,'dpo',run_id))
        print('Found adapters from previous dpo iterations: ', dpo_adapters)
        for adpt in dpo_adapters:
            adapters.append(os.path.join(lora_path,'dpo',run_id,adpt))
    
    return adapters

def format_adapters(adapters: list[str]) -> str:
    return f"[{','.join(adapters)}]"


def to_str(adapters_path: list[str]) -> str:
    return f"[{','.join(adapters_path)}]"


def format_hydra_args(dataset: str, group_id: str, adapters: list[str], run_id: str, model_name: str, iteration: int) -> str:
    """
    Build a properly escaped Hydra override string for dataset, adapters, group ID, run ID, and model name.
    Important: dataset path values containing '=' must be escaped or quoted appropriately.
    """
    escape_dataset = dataset.replace("=", "\\=")
    return (
        f"\"group_id={group_id} run_id={run_id} "
        f"'adapters_paths={to_str(adapters)}' "
        f"'dataset={escape_dataset}' "
        f"iteration={iteration} "
        f"model_config.model_name_or_path={model_name}\""
    )


class JobManager:
    """Handles SLURM job submission and dependency tracking."""

    def __init__(self):
        self.last_job_id = -1

    def submit(self, cmd: str) -> int:
        if self.last_job_id != -1:
            cmd = f"sbatch --dependency=afterok:{self.last_job_id} {cmd}"
        else:
            cmd = f"sbatch {cmd}"
        output = self.run_cmd(cmd)
        self.last_job_id = self.extract_job_id(output)
        return self.last_job_id

    def run_cmd(self, cmd: str) -> str:
        log.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            log.error(result.stderr)
            raise RuntimeError(result.stderr)
        return result.stdout.strip()

    def extract_job_id(self, output: str) -> int:
        match = re.search(r"Submitted batch job (\d+)", output)
        if not match:
            raise ValueError("No job ID found")
        return int(match.group(1))


class Pipeline:
    """Orchestrates the full training + evaluation pipeline using SLURM."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.train_steps = cfg.train_steps.split(" ")
        self.adapters = get_adapters_list(
                            lora_path=cfg.lora_path,
                            run_id=cfg.run_id,
                            start_iter=cfg.start_it
                        )#[] #[f"lora/{cfg.step}/{cfg.model_id}"]
        self.group_id = cfg.group_id
        self.run_id = cfg.run_id #str(uuid.uuid4())[:7] #Need run_id from sft stage
        #self.run_id = '0e5f995'
        #self.adapters.append('lora/sft/0e5f995')
        self.job_mgr = JobManager()
        self.sts_model_path = self.cfg.sts_model.replace("/", "-")
        self.model_name = cfg.model_name.replace("/", "-")

    def base_model_path(self) -> str:
        return (
            f"datasets/{self.cfg.domain}/"
            f"model={self.model_name}_size={self.cfg.size}"
        )

    def output_file_path(self,iter) -> str:
        return f"{self.base_model_path()}/dpo_iter_{iter}"

    def dataset_path(self,iter) -> str:
        if iter == 1:
            return (
                f"{self.base_model_path()}/model={self.sts_model_path}_scored_dpo_{self.cfg.sorting}.parquet"
            )
        else:
            return f"{self.base_model_path()}/dpo_iter_{iter-1}/model={self.sts_model_path}_scored_dpo_{self.cfg.sorting}.parquet"
            
    def run_train_steps(self, iter):
        script = self.cfg.scripts[self.cfg.step]
        dataset = self.dataset_path(iter)
        hydra_args = format_hydra_args(dataset, self.group_id, self.adapters, self.run_id, self.cfg.model_name, iter) #TODO: check hydra args with dpo script
        run_id = self.run_id 

        cmd = (
            f"--export=ALL,WANDB_RUN_ID={run_id} {script} "
            f"--HYDRA_CONFIG {self.cfg.domain} --HYDRA_ARGS {hydra_args}"
        )
        self.job_mgr.submit(cmd)
        self.adapters.append(f"lora/{self.cfg.step}/{run_id}/iteration_{iter}")

    def run_generation(self,iter):
        cmd = (
            f"{self.cfg.scripts.generation} "
            f"--DATASET_PATH {self.cfg.private_path} "
            f"--ADAPTERS_PATHS {','.join(self.adapters)} "
            f"--OUTPUT_PATH {self.output_file_path(iter)} "        #output_path: folder for genrated_public.parquet
            f"--RUN_ID {self.run_id} "
            f"--N_PROMPTS {self.cfg.size_generation} "
            f"--MODEL {self.cfg.model_name}"
        )
        self.job_mgr.submit(cmd)

    def run_score(self,iter):
        cmd = (
            f"{self.cfg.scripts.score} "
            f"--STS_MODEL {self.cfg.sts_model} "                             
            f"--PRIVATE_DATASET {self.cfg.private_path} "
            f"--OUTPUT_PATH {self.output_file_path(iter)} "
            f"--N 4 --WDB_ID {self.run_id} --GROUP_ID {self.group_id} "
            f"--SLURM_GPUS_ON_NODE {self.cfg.tp} "
            f"--MODEL_NAME {self.cfg.model_name} "
        )
        self.job_mgr.submit(cmd)

    def run_filter(self,iter):
        path = f"{self.output_file_path(iter)}/model={self.sts_model_path}_scored.parquet"
        self.job_mgr.submit(f"{self.cfg.scripts.filter} --INPUT_FILE {path}")

    def run_eval(self):
        step = self.train_steps[-1]
        path = f"{self.output_model_path()}/model={self.sts_model_path}_scored_eval.parquet"
        out_path = (
            f"datasets/health/eval/model_outputs/"
            f"model={self.run_id}_size={self.cfg.size}_step={step}"
        )
        eval_gen_cmd = (
            f"{self.cfg.scripts.eval_gen} --DOWNSTREAM_DS_PATH {path} "
            f"--OUTPUT_PATH {out_path} --GROUP_ID {self.group_id}"
        )
        self.job_mgr.submit(eval_gen_cmd)

        eval_pref_cmd = (
            f"{self.cfg.scripts.eval_preference} --MODEL_ID {self.run_id} --STEP {step} "
            f"--SIZE {self.cfg.size} "
            f"--SUFFIX_RUN_NAME {'-'.join(self.train_steps)}-{self.cfg.sorting} "
            f"--GROUP_ID {self.group_id}"
        )
        self.job_mgr.submit(eval_pref_cmd)


@hydra.main(config_path=".", config_name="grid_rl.yaml", version_base="1.3")
def main(cfg: DictConfig):
    pipeline = Pipeline(cfg)
    for iter in range(cfg.start_it,cfg.end_it):
        pipeline.run_train_steps(iter)
        pipeline.run_generation(iter)
        pipeline.run_score(iter)
        pipeline.run_filter(iter)
        #pipeline.run_eval()


if __name__ == "__main__":
    main()
