import os
import random
from typing import List

import pandas as pd
from datasets import load_dataset
from preprocessing.keywords_extraction import KeywordsExtractor as Matcher
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams

import pandas as pd

SAMPLE_SIZE = 1500
RANDOM_SEED = 42
SEED_SIZE = 500
MODEL_NAME = "xz97/AlpaCare-llama2-13b"
GPUS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 2048
OUTPUT_PATH = (
    f"datasets/health/model={MODEL_NAME.replace('/', '-')}_t" f"={TEMPERATURE}_size={SAMPLE_SIZE}"
)
PROMPT_PATH = "datasets/preprocessing/health/prompt.txt"

tqdm.pandas()


#class KeywordExtractor:
#    def __init__(self):
#        self.matcher = QuickUMLS(quickumls_fp=os.getenv("QUICKUMLS_PATH"))
#
#    def extract_keywords(self, text: str) -> str:
#        if pd.isna(text):
#            return ""
#        matches = self.matcher.match(
#            text.strip(),
#            best_match=True,
#            ignore_syntax=False,
#        )
#        return ", ".join([match[0]["term"] for match in matches])

class KeywordExtractor:
    def __init__(self):
        self.matcher = Matcher()

    def extract_keywords(self, text: str) -> str:
        if pd.isna(text):
            return ""
        matches = self.matcher.extract(
            text.strip(),
            best_match=True,
            ignore_syntax=False,
        )
        return ", ".join([match["match"] for match in matches])

class DataProcessor:
    def __init__(
        self,
        note_path: str,
        sample_size: int = SAMPLE_SIZE,
        random_seed: int = RANDOM_SEED,
        seed_size: int = SEED_SIZE,
    ):
        self.sample_size = sample_size
        self.seed_size = seed_size
        random.seed(random_seed)
        self.extractor = KeywordExtractor()

        self.note_path = note_path

    def load_and_sample_dataset(self):
        df = pd.read_csv(self.note_path)  # Adjust path here
        sampled_df = df.sample(n=1000, random_state=42)        
        return sampled_df

    #@staticmethod
    #def reformat_data_sample(sample: dict) -> dict:
    #    if "input" in sample and sample["input"]:
    #        sample["instruction"] = f"{sample['instruction']} {sample['input']}"
    #        del sample["input"]
    #    if "output" in sample:
    #        sample["response"] = sample["output"]
    #        del sample["output"]
    #    return sample

    def process_data(
        self,
        seed_output_path: str = f"{OUTPUT_PATH}/private_seed.parquet",
        gen_output_path: str = f"{OUTPUT_PATH}/private.parquet",
    ):
        
        df = self.load_and_sample_dataset()
        prompt = open(PROMPT_PATH).read()

        df["keywords"] = df["text"].apply(lambda x: self.extractor.extract_keywords(x))
        df["instruction"] = df["keywords"].apply(lambda kws: prompt.replace("{keywords}", kws)).replace("{knowledge_base}", '')
        
        seed_df = df[: self.seed_size].copy()
        gen_df = df[self.seed_size :].copy()

        for path in [seed_output_path, gen_output_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        seed_df.to_parquet(seed_output_path, index=False)
        gen_df.to_parquet(gen_output_path, index=False)

    #@staticmethod
    #def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
    #    prompt = open(PROMPT_PATH).read()
    #    df["new_instruction"] = df.apply(
    #        lambda row: prompt.replace("{keywords}", str(row["instruction_keywords"])).replace(
    #            "{knowledge_base}", str(row["response_keywords"])
    #        ),
    #        axis=1,
    #    )
    #    df["output"] = df["response"]
    #    df["response"] = df["instruction"]
    #    df["instruction"] = df["new_instruction"]
    #    return df[["instruction", "response", "output"]]


def generate_public_seed(
    input_path: str = f"{OUTPUT_PATH}/private_seed.parquet",
    output_path: str = f"{OUTPUT_PATH}/public_seed.parquet",
):
    df = pd.read_parquet(input_path)
    outputs = []
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    hf_model.save_pretrained("model/alpacare")
    llm = LLM(
        model="model/alpacare",
        tensor_parallel_size=GPUS,
    )
    for prompt in tqdm(df["instruction"], desc="Generating responses"):
        response = llm.generate(
            prompt,
            sampling_params=sampling_params,
        )
        outputs.append(output.outputs[0].text for output in response)

    pd.DataFrame({"instruction": df["instruction"], "response": outputs}).to_parquet(output_path)


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_data()
    generate_public_seed()
