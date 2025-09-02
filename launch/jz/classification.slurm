#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:v100:1   # change to >1 if you want multiple GPUs
#SBATCH -N1
#SBATCH -c4
#SBATCH --mem=40000
#SBATCH -p gpuV100  
#SBATCH -w bbs-edsg28-p004

# --- Debugging info ---
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Node rank: $SLURM_NODEID / $SLURM_NNODES"

# --- Conda env setup ---
export PATH="/export/home/cse170020/.user_conda/miniconda/envs/synth-kg/bin:$PATH"
unset CONDA_PREFIX
unset PYTHONPATH
unset CONDA_DEFAULT_ENV

echo "Using python: $(which python)"
python --version

# --- Parse input arguments ---
HYDRA_CONFIG="health.yaml"
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --HYDRA_ARGS)   HYDRA_ARGS="$2"; shift ;;
  *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done
eval set -- "$HYDRA_ARGS"

export WANDB_MODE=offline

# --- Launcher: python (1 GPU) vs torchrun (multi-GPU) ---
if [ "$SLURM_GPUS_ON_NODE" -gt 1 ]; then
    echo "Launching with torchrun on $SLURM_GPUS_ON_NODE GPUs..."
    torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE \
      training_steps/classification/train.py "$@"
else
    echo "Launching with python on 1 GPU..."
    python training_steps/classification/train.py "$@"
fi
