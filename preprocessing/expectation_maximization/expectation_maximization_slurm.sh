#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH -N1-1
#SBATCH -c1
#SBATCH --mem=40000
#SBATCH -w bbs-edsg28-p012

#debug
# Set PATH manually to target the environment's bin directory
export PATH="/export/home/cse170020/.user_conda/miniconda/envs/aphp_env_39/bin:$PATH"

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

export PYTHONPATH="$PWD"

python3 preprocessing/expectation_maximization/expectation_maximization.py ../../data/crh_omop_2024/all/train.csv ../ --vocab_path ../counter_1gram.pkl --iters 50