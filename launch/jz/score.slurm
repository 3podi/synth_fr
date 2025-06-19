#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH -N1-1
#SBATCH -c4
#SBATCH --mem=40000
#SBATCH -p gpuV100
#SBATCH -w bbs-edsg28-p004

# Initialize variables with default values
STS_MODEL=""
PRIVATE_DATASET=""
N=4
OUTPUT_PATH=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --STS_MODEL)
    STS_MODEL="$2"
    shift
    ;;
  --PRIVATE_DATASET)
    PRIVATE_DATASET="$2"
    shift
    ;;
  --N)
    N="$2"
    shift
    ;;
  --OUTPUT_PATH)
    OUTPUT_PATH="$2"
    shift
    ;;
  --WDB_ID)
    WDB_ID="$2"
    shift
    ;;
  --GROUP_ID)
    GROUP_ID="$2"
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

#module load cuda # Adjust to the appropriate CUDA version

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
pip --version

export HF_HUB_OFFLINE=1
echo "$OUTPUT_PATH"
echo "$PRIVATE_DATASET"
python training_steps/score/run.py \
  --sts_model "$STS_MODEL" \
  --public_dataset "$OUTPUT_PATH" \
  --private_dataset "$PRIVATE_DATASET" \
  --tp "$SLURM_GPUS_ON_NODE" \
  --output_path "$OUTPUT_PATH" \
  --n "$N" \
  --wdb_id "$WDB_ID" \
  --group_id "$GROUP_ID"
