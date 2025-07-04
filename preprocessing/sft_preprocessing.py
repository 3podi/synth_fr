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

OUTPUT_PATH = (
    f"datasets/health/sft"
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
    

def process_batch_keywords(batch, list_definitions):
    extractor = KeywordExtractor(list_definitions,database_dir='prova')
    #unique_keys = list(dict.fromkeys([extractor.extract_keywords(text) for text in batch['text']]))
    kws = [extractor.extract_keywords(text) for text in batch['text']]
    #kws = list(dict.fromkeys(kws))
    return {"keywords": kws}

def process_batch_instruction(batch, prompt):
    return {
        "instruction": [
            prompt.replace("{keywords}", kws).replace("{knowledge_base}", "") 
            for kws in batch['keywords']
        ]
    }

class DataProcessor:
    def __init__(self, parquet_dir, list_definitions):

        self.extractor = KeywordExtractor(list_definitions=list_definitions)
        self.list_definitions = list_definitions
        self.parquet_dir = parquet_dir
        
    def load_and_sample_dataset(self, parquet_path):
        df = load_dataset("parquet", data_files=parquet_path, split='train')
        return df

    def process_data(self, output_path=OUTPUT_PATH, prompt_path='prompt.txt'):
        
        parquet_files = os.listdir(self.parquet_dir)
        parquet_files = [pf for pf in parquet_files if pf.endswith('.parquet')]
        print(parquet_files)
        prompt = open(PROMPT_PATH).read()
        
        os.makedirs(output_path, exist_ok=True)

        for parquet in parquet_files:
            print('Processing ', parquet)
            
            parquet_path = os.path.join(self.parquet_dir,parquet)
            df = self.load_and_sample_dataset(parquet_path)

            # Extract keywords
            if "keywords" not in df.column_names:
                df = df.map(
                    process_batch_keywords,
                    batched=True,
                    batch_size=1000,
                    num_proc=4,
                    fn_kwargs={'list_definitions': self.list_definitions},
                    #load_from_cache_file=False,
                    desc="Extracting keywords"
                )
                
            if "instruction" not in df.column_names:
                df = df.map(
                    process_batch_instruction,
                    batched=True,
                    batch_size=1000,
                    num_proc=4,
                    fn_kwargs={'prompt': prompt},
                    #load_from_cache_file=False,
                    desc="Making instructions"
                )

            df = df.rename_column("text", "response")

            save_path = os.path.join(output_path, parquet)
            df.to_pandas().to_parquet(save_path, index=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= 'Setting note path')
    parser.add_argument('parquet_dir', type=str, help='Name of hf dataset')    
    parser.add_argument('strings_path', type=str, help='Path to the strings for similarity matching')    
    args = parser.parse_args()

    with open(args.strings_path, 'rb') as file:
        definitions = pickle.load(file)
    definitions = definitions.keys()
    
    processor = DataProcessor(args.parquet_dir, list_definitions=definitions)
    processor.process_data()