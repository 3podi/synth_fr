import os
import random
from typing import List

import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from vllm import LLM, SamplingParams

MODEL_NAME = "meta-llama/llama-2-7b-hf" #"xz97/AlpaCare-llama2-13b"
GPUS = 1
SAMPLE_SIZE = 1500
TEMPERATURE = 0.7
MAX_TOKENS = 2048
OUTPUT_PATH = (
    f"datasets/health/model={MODEL_NAME.replace('/', '-')}_t" f"={TEMPERATURE}_size={SAMPLE_SIZE}"
)

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
        #model="model/alpacare",
        model=MODEL_NAME,
        tensor_parallel_size=GPUS,
    )
    #for prompt in tqdm(df["instruction"], desc="Generating responses"):
    #    response = llm.generate(
    #        prompt,
    #        sampling_params=sampling_params,
    #    )
    #    outputs.append(output.outputs[0].text for output in response)
    prompts = df["instruction"].tolist()
    response = llm.generate(
        prompts,
        sampling_params=sampling_params,
    )
    outputs.append(output.outputs[0].text for output in response)
    pd.DataFrame({"instruction": df["instruction"], "response": outputs}).to_parquet(output_path)


if __name__ == "__main__":
    
    generate_public_seed()
