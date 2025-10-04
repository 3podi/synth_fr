# Synth-FR: Synthetic French Medical Report Generation

## Overview
**Synth-FR** is a framework for generating, training, and evaluating large language models on **synthetic French medical reports**.  
The project’s primary goal is to create a high-quality synthetic corpus for tasks such as **ICD-10 coding** using modern fine-tuning and reinforcement learning techniques.

The Synth-FR pipeline consists of several key stages:

1. **Supervised Fine-Tuning (SFT)**  
   * A small **seed set of French medical reports** is used to fine-tune a base small generator model.

2. **Generation**  
   * Using linical keywords (e.g., extracted from private documents inside a hospital), the generator produces multiple synthetic candidate reports.  
   * These reports mimic the structure and terminology of real discharge summaries, without leaking personal identifiers.  

3. **Scoring**  
   * An evaluator model compares real vs. synthetic reports using only floating-point similarity scores.  
   * This ensures no raw text ever leaves the private side of the pipeline, maintaining strict privacy guarantees.  

4. **Alignment with Direct Preference Optimization (DPO)**  
   * Synthetic candidate pairs are ranked using the evaluator.  
   * The generator is aligned via DPO, improving report quality and realism across successive iterations.  

5. **Evaluation on Downstream Tasks**  
   * Synthetic datasets are tested on downstream task:  
     - **ICD-10 coding** (multi-label classification).  

This framework is designed for use on **HPC clusters** with SLURM job scheduling as well as local experiments.

---

**Resources on Hugging Face:**  
All models and datasets for this project are available in this [HF collection](https://huggingface.co/collections/3podi/synth-fr-68e03a5bd90eab12babd4ecc), including:

- **4B Qwen model** ready to use for reports generation.
- **LoRA modules** for each DPO step, allowing fine-tuning at different stages of the pipeline.  
- **Step-specific datasets** with 50K samples per DPO step.  
- **MedGemma French dataset** with 300K samples, useful for distilling a model to generate reports from keywords.  

These resources make it easy to experiment with any stage of the training and inference pipeline.

---

## Structure

```
synth_fr-main/
  ├── environment_preprocessing.yml # Preprocessing / keywords extraction environment
  ├── environment.yml               # Training/generation conda environment definition
  ├── setup.py                      # Set up dataset and download from hf
  ├── grid_*.py/.yaml               # Experiment pipelines scipts and configs (SFT, RL, etc.)
  ├── launch/jz/                    # SLURM job scripts
  ├── training_steps/               # Training/generation scripts
  │   ├── sft/                      # Pre-Training (SFT)
  │   ├── generation/               # Reports Generation
  │   ├── score/                    # Reports scoring (Cosine similarity, LLM as a Judge)
  │   ├── filter/                   # Building DPO dataset
  │   ├── alignment/                # Alignment methods (DPO)
  │   └── classification/           # Classification training script
  └── README.md                     # Project documentation
```

---

## Usage

### Environment Setup

There are **two separate Conda environments** for Linux systems depending on the stage of the pipeline:

1. **Dataset Preparation Environment**  
   Defined in `environment_preprocessing.yml`.  
   This environment is used for **preprocessing tasks**, such as keyword extraction and dataset preparation.

   ```bash
   conda env create -f environment_preprocessing.yml -n synth-fr-prep
   python -m spacy download fr_core_news_sm
   conda activate synth-fr-prep

2. **Pipeline Environment**
   Defined in `environment.yml` and is used for running anything else.

   ```bash
   conda env create -f environment.yml -n synth-fr
   conda activate synth-fr


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
* By default 2 files required for extracting keywords / generation are downloaded from HF.


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

---
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
  The important parameters (private_path, model_name, sts_model, size_sft) are the same as for SFT.


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

We ran downstream ICDD-10 classification task on different training synthetic dataset sizes (`10k`, `20k`, `50k`) and different numbers of top-k most frequent labels in the dataset (`20`, `50`, `100`).  
For each setting, the **[answerdotai/ModernBERT](https://huggingface.co/answerdotai/ModernBERT)** encoder was fine-tuned for classification using the following command:

```bash
python training_steps/classification/train.py --config training_steps/classification/config.yaml
```

The experiments compare the baseline model (Step0) with models further refined using Direct Preference Optimization (DPO) for up to three iterations.
Performance is reported in terms of macro F1 on a curated private evaluation set that is distinct from the training and synthetic documents, ensuring the results reflect transferability beyond synthetic data.
The final column highlights the relative improvement of DPO3 over Step0.


Results on 10k dataset

| top_k | Step0  | DPO1       | DPO2   | DPO3       | DPO3 vs Step0 (Rel. Impr.) |
|-------| ------ | ---------- | ------ | ---------- | -------------------------- |
| 20    | 0.3893 | **0.4223** | 0.4215 | 0.4180     | +7.4%                      |
| 50    | 0.2498 | 0.2937     | 0.3024 | **0.3171** | **+26.9%**                 |
| 100   | 0.1747 | 0.1965     | 0.2020 | **0.2384** | **+36.5%**                 |

Results on 20k dataset

| top_k | Step0  | DPO1       | DPO2   | DPO3       | DPO3 vs Step0 (Rel. Impr.) |
|-------| ------ | ---------- | ------ | ---------- | -------------------------- |
| 20    | 0.4383 | **0.4569** | 0.4476 | 0.4474     | +2.1%                      |
| 50    | 0.2988 | 0.3264     | 0.3341 | **0.3473** | **+16.2%**                 |
| 100   | 0.1954 | 0.2298     | 0.2096 | **0.2849** | **+45.8%**                 |

Results on 50k dataset

| top_k | Step0  | DPO1       | DPO2       | DPO3       | DPO3 vs Step0 (Rel. Impr.) |
|-------| ------ | ---------- | ---------- | ---------- | -------------------------- |
| 20    | 0.4450 | **0.4614** | 0.4612     | 0.4550     | +2.3%                      |
| 50    | 0.3256 | 0.3507     | **0.3845** | 0.3828     | **+17.6%**                 |
| 100   | 0.2302 | 0.2675     | 0.2995     | **0.3159** | **+37.2%**                 |

Findings
- Repeated DPO iterations improve results, with the biggest relative gains at **higher top-k (100)**.  
- At smaller top-k (20), DPO1 already provides strong results, sometimes even outperforming later iterations.  


---
### Report Generation
The `training_steps/generation/` scripts should be preferred for **fast report generation** as they rely on the [vLLM engine](https://github.com/vllm-project/vllm), 
which provides highly optimized inference for large language models.  
This allows generating large batches of synthetic medical reports efficiently compared to standard Hugging Face pipelines.


## Quickstart: Load from Hugging Face

After training and uploading the model to [🤗 Hugging Face Hub](https://huggingface.co/), you can load it in Python and generate synthetic medical reports:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "3podi/Qwen3-4B-Instruct-2507-synth-fr"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype='bfloat16')

# Build a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Example prompt
prompt = """### Instruction
Vous agissez en tant qu'IA médicale spécialisée. Votre tâche consiste à rédiger un rapport d'hospitalisation fictif complet et réaliste. Le document doit présenter une évolution clinique cohérente avec une terminologie médicale précise, tout en respectant scrupuleusement la séquence imposée des mots-clés. Retournez uniquement le rapport.
### Keywords
{keywords}
### Output"""

keywords = ", ".join(["Infection virale des voies respiratoires", "Maladie inflammatoire des voies respiratoires"])
prompt = prompt.replace("{keywords}", keywords)

# Generate
outputs = generator(prompt, max_length=3000, num_return_sequences=1, do_sample=True, top_p=0.9)
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



### Example Generated Synthetic Report

```text
### Motif d’hospitalisation
Hospitalisation pour prise en charge d’une fracture ouverte du pouce gauche suite à un accident domestique. Contusion associée du doigt index gauche avec lésion de l’ongle.

### Antécédents
Patiente de 42 ans, sans antécédents médicaux ou chirurgicaux notables. Allergie connue à la pénicilline (éruption cutanée).

### Mode de vie
Non-fumeuse, consommation d’alcool occasionnelle (1 à 2 verres de vin par semaine). Profession : enseignante. Pratique une activité physique modérée (marche, yoga). Vit en couple, sans enfants. Situation sociale stable.

### Histoire de la maladie
La patiente rapporte être tombée dans son domicile le 15/07/2024, entraînant une chute sur la main gauche. Elle décrit une douleur immédiate et intense au niveau du pouce gauche, associée à une plaie cutanée. Elle s’est rendue aux urgences le même jour.

### Examen clinique
**Constantes vitales (15/07/2024) :** TA 120/75 mmHg, FC 88 bpm, FR 16/min, SpO2 98% AA, T° 36.8°C.  
**Examen physique (15/07/2024) :**
* **Inspection :** Plaie d’environ 2 cm de long, située sur la face radiale de la base du pouce gauche, avec saignement actif modéré. Déformation du pouce gauche. Contusion du doigt index gauche avec hématome sous-unguéal et avulsion partielle de l’ongle.
* **Palpation :** Mobilité douloureuse et crépitante au niveau de la base du pouce gauche. Douleur à la palpation du doigt index gauche.
* **Neurovasculaire :** Sensibilité conservée au niveau du pouce et de l’index gauche. Pouls radial et ulnaire palpables. Absence de déficit moteur majeur initial.
* **Autres :** Absence de signe de syndrome compartimental.

### Examens complémentaires
* **Radiographie du poignet et de la main gauche (15/07/2024) :** Fracture ouverte comminutive de la base du premier métacarpien gauche, avec déplacement important des fragments. Fracture non déplacée du phénix du doigt index gauche.
* **Biologie (15/07/2024) :** NFS : GB 12 500/mm³, Hémoglobine 13.5 g/dL, Plaquettes 350 000/mm³. CRP : 45 mg/L. Ionogramme et fonction rénale normaux.
* **ECG (15/07/2024) :** Rythme sinusal, fréquence 88 bpm, sans anomalies significatives.

### Évolution pendant l'hospitalisation
**Prise en charge diagnostique :** Diagnostic confirmé de fracture ouverte du pouce gauche et contusion du doigt index gauche avec lésion de l’ongle. Légère leucocytose et CRP élevée.  
**Conclusions des examens complémentaires :** Fracture comminutive nécessitant prise en charge chirurgicale. Fracture de l’index sans déplacement.  
**Pathologies finales retenues :** Fracture ouverte comminutive de la base du premier métacarpien gauche. Contusion du doigt index gauche avec hématome sous-unguéal et avulsion partielle de l’ongle.  
**Prise en charge thérapeutique :**
* **15/07/2024 :** Nettoyage et débridement chirurgical de la plaie sous anesthésie loco-régionale. Pansement chirurgical. Antibiothérapie IV par Céfazoline 2g/jour. Antalgiques (Paracétamol 1g x3/jour, Tramadol 100mg si besoin).
* **17/07/2024 :** Ostéosynthèse par voie ouverte (fixation par plaque et vis). Fermeture de la plaie. Pansement chirurgical. Antibiothérapie IV pendant 48h puis orale (Amoxicilline/Ac. clavulanique 1g x3/jour).
* **Suivi :** Surveillance clinique quotidienne, pansements réguliers, kinésithérapie débutée le 18/07/2024.

**Critères de sortie :**
* Absence de fièvre.
* Pansement propre et sec.
* Douleur contrôlée par antalgiques oraux.
* Bonne mobilité des doigts et du poignet.
* Compréhension des consignes de suivi et de rééducation.

### Conclusion
Patiente de 42 ans admise pour fracture ouverte du pouce gauche et contusion du doigt index gauche suite à une chute. Prise en charge chirurgicale avec ostéosynthèse et débridement. Évolution favorable. Sortie le 20/07/2024 avec suivi ambulatoire prévu.
```
