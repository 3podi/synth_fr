import os
import random
from typing import List

import sys
import pandas as pd
from keywords_extraction import KeywordsExtractor as Matcher
from tqdm import tqdm

import argparse
import pickle

from datasets import load_dataset

SAMPLE_SIZE = 1500
RANDOM_SEED = 42
SEED_SIZE = 500
#MODEL_NAME = "google/medgemma-4b-it"#"mistralai/Mistral-7B-v0.1" #"meta-llama/llama-2-7b-hf" #"xz97/AlpaCare-llama2-13b"
#GPUS = 1
#TEMPERATURE = 0.7
#MAX_TOKENS = 2048
OUTPUT_PATH = (
    f"datasets/health/"
)
PROMPT_PATH = "datasets/preprocessing/health/prompt.txt"

tqdm.pandas()

class KeywordExtractor:
    def __init__(self, list_definitions, database_dir=None):
        self.matcher = Matcher(list_definitions=list_definitions, database_dir=database_dir)

    def extract_keywords(self, text: str) -> str:
        #if pd.isna(text):
        #    return ""
        matches = self.matcher.extract(text)
        return ", ".join([match["match"] for match in matches])
    

def process_batch_keywords2(batch, list_definitions):
    extractor = KeywordExtractor(list_definitions,database_dir='prova')
    return {"keywords": [extractor.extract_keywords(text) for text in batch['text']]}

def process_batch_instruction(batch, prompt):
    return {
        "instruction": [
            prompt.replace("{keywords}", kws).replace("{knowledge_base}", "") 
            for kws in batch['keywords']
        ]
    }

class DataProcessor3:
    def __init__(self, dataset_name, list_definitions, random_seed=42, sample_size=SAMPLE_SIZE):
        random.seed(random_seed)

        self.extractor = KeywordExtractor(list_definitions=list_definitions)
        self.dataset_name = dataset_name
        self.list_definitions = list_definitions
        self.sample_size = sample_size

    def load_and_sample_dataset(self):
        df = load_dataset(self.dataset_name)
        df = df['fr']
        
        df = df.shuffle(seed=42)
        df = df.select(range(self.sample_size))
        return df

    def process_data(self, output_path='output', prompt_path='prompt.txt'):
        output_path = os.path.join(output_path, self.dataset_name.replace('/', '-'))
        output_path = os.path.join(output_path, 'sft.parquet')

        df = self.load_and_sample_dataset()
        prompt = open(PROMPT_PATH).read()


        # Pass minimal objects to map, avoid passing self or complex objects
        df = df.map(
            process_batch_keywords2,
            batched=True,
            batch_size=1000,
            num_proc=4,
            fn_kwargs={'list_definitions': self.list_definitions},
            #load_from_cache_file=False,
            desc="Extracting keywords"
        )

        df = df.map(
            process_batch_instruction,
            batched=True,
            batch_size=1000,
            num_proc=4,
            fn_kwargs={'prompt': prompt},
            #load_from_cache_file=False,
            desc="Making instructions"
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_pandas().to_parquet(output_path, index=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= 'Setting note path')
    parser.add_argument('dataset_name', type=str, help='Name of hf dataset')    
    parser.add_argument('strings_path', type=str, help='Path to the strings for similarity matching')    
    args = parser.parse_args()

    with open(args.strings_path, 'rb') as file:
        definitions = pickle.load(file)
    definitions = definitions.keys()
    
    processor = DataProcessor3(args.dataset_name, list_definitions=definitions)
    processor.process_data()
