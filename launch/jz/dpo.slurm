#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH -N1-1
#SBATCH -c4
#SBATCH --mem=40000
#SBATCH -p gpuV100
#SBATCH -w bbs-edsg28-p004

# Initialize variables with default values
HYDRA_CONFIG="default.yaml"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --HYDRA_CONFIG)
    HYDRA_CONFIG="$2"
    shift
    ;;
  --HYDRA_ARGS)
    HYDRA_ARGS="$2"
    shift
    ;;
  *)
    echo "Unknown parameter passed: $1"
    exit 1
    ;;
  esac
  shift
done

#module purge
#module load arch/h100
#module load miniforge/24.9.0
#conda activate synth-kg

#debug
# Set PATH manually to target the environment's bin directory
export PATH="/export/home/cse170020/.user_conda/miniconda/envs/synth-kg/bin:$PATH"

# Optional: unset interfering Conda env vars
unset CONDA_PREFIX
unset PYTHONPATH
unset CONDA_DEFAULT_ENV

# Debug output
echo "PATH: $PATH"
echo "Using python: $(which python)"
python --version
echo "Using pip: $(which pip)"

export WANDB_MODE=offline
eval set -- "$HYDRA_ARGS"
python training_steps/alignment/dpo.py --config-name="$HYDRA_CONFIG" "$@"
