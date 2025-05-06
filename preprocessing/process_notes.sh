#!/bin/bash
#SBATCH -t 3:00:00
#SBATCH -N1-1
#SBATCH -c4
#SBATCH --mem=10000
#SBATCH -w bbs-edsg28-p012

# Load Conda environment
source $HOME/.user_conda/miniconda/etc/profile.d/conda.sh
conda activate $HOME/.user_conda/miniconda/envs/aphp_env_39

#pip list

# Define directories for input and output
notePath=$HOME/data/crh_omop_2024/test_1000
myRep=Riccardo_T
myDicts=$HOME/$myRep/dictionaries  # Path to your source data
myDestPath=$HOME/$myRep  # Path to copy data in scratch space
myOutPut=training$SLURM_JOBID  # Output directory (including job ID for uniqueness)

# Set paths for the input data and output directory in scratch
input_path=$notePath/train.csv
output_path=$myDestPath/slurm_outputs/$myOutPut
#mkdir -p $output_path
output_path= $output_path/results_matching.csv

# Run the Python script with the input and output paths as command-line arguments
#python /export/home/cse170020/Riccardo_T/prova_slurm.py $input_path $myDicts/aphp_final.pkl $output_path 
#python /export/home/cse170020/Riccardo_T/synth_fr/preprocessing/note_processor_multi.py $input_path $myDicts/aphp_final.pkl $output_path

#python preprocessing/note_processor_multi.py $input_path $myDicts/aphp_final.pkl $output_path
python preprocessing/note_processor_multi.py
