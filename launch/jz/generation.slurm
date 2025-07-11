#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH -N1-1
#SBATCH -c4
#SBATCH --mem=40000
#SBATCH -p gpuV100  
#SBATCH -w bbs-edsg28-p006

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

# Initialize variables with default values
DATASET_PATH="default_name"
#MODEL="meta-llama/llama-2-7b-hf"
#MODEL="mistralai/Mistral-7B-v0.1"
ADAPTERS_PATHS="default_path"
OUTPUT_PATH="default_path"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --DATASET_PATH)
    DATASET_PATH="$2"
    shift
    ;;
  --MODEL)
    MODEL="$2"
    shift
    ;;
  --ADAPTERS_PATHS)
    ADAPTERS_PATHS="$2"
    shift
    ;;
  --OUTPUT_PATH)
    OUTPUT_PATH="$2"
    shift
    ;;
  --RUN_ID)
    RUN_ID="$2"
    shift
    ;;
  --N_PROMPTS)
    N_PROMPTS="$2"
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

# Run your processing script
#MERGE_ID=$(printf "%06d" $((RANDOM % 1000000)))
#MERGE_OUTPUT_PATH=./lora/merge-${MERGE_ID}
MERGE_OUTPUT_PATH=./lora/merge/${RUN_ID}

CUDA_VISIBLE_DEVICES=0 python training_steps/generation/merge_adapters.py \
  --model "$MODEL" \
  --adapters "$ADAPTERS_PATHS" \
  --output_path "$MERGE_OUTPUT_PATH"

python training_steps/generation/run.py --dataset "$DATASET_PATH" \
  --model "$MERGE_OUTPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_prompts "$N_PROMPTS" \
  #--tp "$SLURM_GPUS_ON_NODE" \
  #--pp "$SLURM_NNODES"
