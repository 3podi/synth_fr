import os
import random
from typing import List

import sys
import pandas as pd
from keywords_extraction import KeywordsExtractor as Matcher
from tqdm import tqdm

import argparse
import pickle

SAMPLE_SIZE = 1500
RANDOM_SEED = 42
SEED_SIZE = 500
MODEL_NAME = "meta-llama/llama-2-7b-hf" #"xz97/AlpaCare-llama2-13b"
GPUS = 1
TEMPERATURE = 0.7
MAX_TOKENS = 2048
OUTPUT_PATH = (
    f"datasets/health/model={MODEL_NAME.replace('/', '-')}_t" f"={TEMPERATURE}_size={SAMPLE_SIZE}"
)
PROMPT_PATH = "datasets/preprocessing/health/prompt.txt"

tqdm.pandas()

class KeywordExtractor:
    def __init__(self, list_definitions):
        self.matcher = Matcher(list_definitions=list_definitions)

    def extract_keywords(self, text: str) -> str:
        #if pd.isna(text):
        #    return ""
        matches = self.matcher.extract(text)
        return ", ".join([match["match"] for match in matches])

class DataProcessor:
    def __init__(
        self,
        note_path: str,
        list_definitions: List[str],
        sample_size: int = SAMPLE_SIZE,
        random_seed: int = RANDOM_SEED,
        seed_size: int = SEED_SIZE,
    ):
        self.sample_size = sample_size
        self.seed_size = seed_size
        random.seed(random_seed)
        self.extractor = KeywordExtractor(list_definitions=list_definitions)

        self.note_path = note_path

    def load_and_sample_dataset(self):
        df = pd.read_csv(self.note_path)  # Adjust path here
        sampled_df = df.sample(n=self.sample_size, random_state=42)        
        return sampled_df


    def process_data(
        self,
        seed_output_path: str = f"{OUTPUT_PATH}/private_seed.parquet",
        gen_output_path: str = f"{OUTPUT_PATH}/private.parquet",
    ):
        
        df = self.load_and_sample_dataset()
        prompt = open(PROMPT_PATH).read()

        df["keywords"] = df["text"].apply(lambda x: self.extractor.extract_keywords(x))
        df["instruction"] = df["keywords"].apply(
            lambda kws: prompt.replace("{keywords}", kws).replace("{knowledge_base}", '')
        )
        
        seed_df = df[: self.seed_size].copy()
        gen_df = df[self.seed_size :].copy()

        for path in [seed_output_path, gen_output_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        seed_df.to_parquet(seed_output_path, index=False)
        gen_df.to_parquet(gen_output_path, index=False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= 'Setting note path')
    parser.add_argument('note_path', type=str, help='Path to the notes')    
    parser.add_argument('strings_path', type=str, help='Path to the strings for similarity matching')    
    args = parser.parse_args()

    with open(args.strings_path, 'rb') as file:
        definitions = pickle.load(file)
    definitions = definitions.keys()
    
    processor = DataProcessor(args.note_path, list_definitions=definitions)
    processor.process_data()