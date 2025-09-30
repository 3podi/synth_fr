# Synth-FR: Synthetic French Medical Report Generation

## Overview
**Synth-FR** is a framework for generating, training, and evaluating large language models on **synthetic French medical reports**.  
The project’s primary goal is to create a high-quality synthetic corpus for tasks such as **ICD-10 coding** using modern fine-tuning and reinforcement learning techniques.

The pipeline involves pretraining on a seed of privacy-free documents, and then generating new reports from medical keywords extracted from, ideally, even private documents. For example, inside a hospital where private documents exist, they can be used to provide feedback to further improve generation without exposing sensitive data.

The repo includes pipelines for:
- **Supervised Fine-Tuning (SFT)**
- **Generation and scoring/filtering reports**
- **Direct Preference Optimization (DPO)**
- **Evaluation on downstream Classification tasks**

This framework is designed for use on **HPC clusters** with SLURM job scheduling as well as local experiments.

---

## Structure

```
synth_fr-main/
  ├── xxxx.yml                # Preprocessing / keywords extraction environment
  ├── environment.yml         # Training/generation conda environment definition
  ├── grid_*.py/.yaml         # Experiment pipelines scipts and configs (SFT, RL, etc.)
  ├── launch/jz/              # SLURM job scripts
  ├── training_steps/         # Training/generation scripts
  │   ├── sft/                # Pre-Training (SFT)
  │   ├── generation/         # Reports Generation
  │   ├── score/              # Reports scoring (Cosine similarity, LLM as a Judge)
  │   ├── filter/             # Building DPO dataset
  │   ├── alignment/          # Alignment methods (DPO)
  │   └── classification/     # Classification training script
  └── README.md               # Project documentation
```

---

## Usage

### Environment Setup

There are **two separate Conda environments** depending on the stage of the pipeline:

1. **Dataset Preparation Environment**  
   Defined in `xxxx.yml`.  
   This environment is used for **preprocessing tasks**, such as keyword extraction and dataset preparation.

   ```bash
   conda env create -f xxxx.yml
   conda activate synth-fr-prep

2. **Pipeline Environment**
   Defined in `environment.yml` and is used for **running the main pipelines**.

   ```bash
   # Create and activate the pipeline environment
   conda env create -f environment.yml
   conda activate synth-fr

### Preparing the Dataset for Supervised Fine-Tuning (SFT)

Before running SFT, your dataset needs to be processed to extract keywords and format instructions correctly. The provided script automates this processing.

#### Dataset Format

* Input dataset should be in **Parquet** format.
* Each file should contain a text column (or `input`/`response`) to extract keywords from.
* Example structure before processing:

```
| text                                     |
|-----------------------------------------|
| Patient presents with fever and cough.  |
| History of hypertension and diabetes.   |
```

After processing, the dataset will have columns like:

```
| instruction                                           | keywords                  | response                          |
|------------------------------------------------------|---------------------------|-----------------------------------|
| ...prompt with {keywords} replaced...                | Infection, Respiratory ...| Patient presents with fever ...    |
```

#### Running the Processing Script

Use the following command to process your dataset:

```bash
python preprocessing/sft_preprocessing.py \
    --parquet_dir /path/to/parquet_files \
    --strings_path /path/to/definitions.pkl \
    [--random_extraction]
```

* `--parquet_dir`: Directory containing your raw Parquet dataset files.
* `--strings_path`: Path to the pickled definitions (required for extracting keywords).
* `--random_extraction`: Optional. If set, 50% of entries use extracted keywords, 50% use definitions provided with the input text.

If `--random_extraction` is used, every input should have a 'definitions' column which represent the keywords to use.

#### Output

* Processed files will be saved to:

```
datasets/health/sft/
    ├─ processed_file_1.parquet
    ├─ processed_file_2.parquet
    └─ ...
```

These processed Parquet files can then be directly fed into the SFT training script.



### Setup Dataset Repository

You can create the dataset folder structure and copy the seed files using the setup script. Replace the paths with your own files and model name:

```bash
python setup.py \
    --model_name <hugging-face-model-path> \
    --private_seed /path/to/private_seed.parquet \
    --public_seed /path/to/public_seed.parquet \
    --dataset_size <dataset_size>
```

**Note:**

* `private_seed` refers to documents that could contain private information.
* `public_seed` refers to privacy-free documents that can be safely used for supervised fine-tuning (SFT).


### Training with SLURM (HPC clusters)

Use the **grid scripts** together with SLURM submission files for large-scale training:

- **Supervised Fine-Tuning (SFT)**
  ```bash
  python grid_sft.py --config grid_sft.yaml
  ```
* `private_path`: Path to the private seed dataset (used for generating sequences).  
* `model_name`: Hugging Face model name or local model path used for training.  
* `sts_model`: LLM used as a judge model for scoring and evaluation.  
* `size_sft`: Number of samples to use for SFT (can be smaller than total dataset size).    


- **Direct Preference Optimization (DPO)**
  ```bash
  python grid_rl.py --config grid_rl.yaml
  ```

Other jobs in `launch/jz/` include:
- `generation.slurm`
- `score.slurm`
- `filter.slurm`
- `classification.slurm`

---

### Training on Local Machines
For debugging or smaller experiments, use the `*_local.py` scripts:

- **SFT**
  ```bash
  python grid_sft_local.py --config grid_sft_local.yaml
  ```

- **DPO / RL**
  ```bash
  python grid_rl_local.py --config grid_rl_local.yaml
  ```

---

### Classification Training
Run downstream classification fine-tuning:
```bash
python training_steps/classification/train.py --config training_steps/classification/config.yaml
```

---
### Report Generation
The `training_steps/generation/` scripts should be preferred for **fast report generation** as they rely on the [vLLM engine](https://github.com/vllm-project/vllm), 
which provides highly optimized inference for large language models.  
This allows generating large batches of synthetic medical reports efficiently compared to standard Hugging Face pipelines.


## Quickstart: Load from Hugging Face

After training and uploading the model to [🤗 Hugging Face Hub](https://huggingface.co/), you can load it in Python and generate synthetic medical reports:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "3podi/xxxxx"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Build a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Example prompt
prompt = "### Instruction
Vous agissez en tant qu'IA médicale spécialisée. Votre tâche consiste à rédiger un rapport d'hospitalisation fictif complet et réaliste. Le document doit présenter une évolution clinique cohérente avec une terminologie médicale précise, tout en respectant scrupuleusement la séquence imposée des mots-clés. Retournez uniquement le rapport.
### Keywords
{keywords}
### Output"

keywords = ", ".join(["Infection virale des voies respiratoires", "Maladie inflammatoire des voies respiratoires"])
prompt = prompt.replace("{keywords}", keywords)

# Generate
outputs = generator(prompt, max_length=200, num_return_sequences=1, do_sample=True, top_p=0.9)
print(outputs[0]["generated_text"])
```

### vLLM Inference

You can generate synthetic medical reports using vLLM with the provided script. Replace the paths and model name as needed:

```bash
python tranining_scripts/generation/random_gen.py \
    --model <hugging-face-model-or-local-path> \
    --csv_path /path/to/csv_with_codes.csv \
    --output_path /path/to/save/output_folder \
    --num_samples 10 \
    --max_codes 5 \
    --max_kws 1 \
    --tp 1 \
    --pp 1
```

### Launching the SLURM Script

You can submit the SLURM script to run large-scale generation on an HPC cluster. Replace the arguments with your dataset, model, and output paths as needed:

```bash
sbatch launch/jz/random_gen.slurm \
    --DATASET_PATH /path/to/dataset \
    --MODEL <hugging-face-model-or-local-path> \
    --OUTPUT_PATH /path/to/save/output \
    --CSV_PATH /path/to/codes.csv \
    --RAND_NUM_SAMPLES 100 \
    --RAND_MAX_CODES 5 \
    --RAND_MAX_KWS 1
```



> **Note:** On [Hugging Face](https://huggingface.co/), you can find:  
> - **LoRA modules** for every DPO step  
> - **Datasets of 50K samples** corresponding to each DPO step  
>
> These resources allow you to experiment with any stage of the pipeline.

