import argparse
import os
import random

import pandas as pd
from vllm import LLM, SamplingParams
from typing import List

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output parquet file"
    )
    parser.add_argument("--model", type=str, required=True, help="Name or path of the model to use")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to input dataset parquet file"
    )

    parser.add_argument(
        "--num_prompts",
        type=int,
        default=100,
        help="Number of different prompts to generate sequences",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=4,
        help="Number of different sequences to generate per prompt",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Number of GPUs to use for inference",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Number of node to use for inference",
    )
    return parser.parse_args()


def generate_responses(model, prompts, num_sequences):
    all_responses = []
    all_temperature = []
    all_top_p = []
    all_top_k = []
    
    tokenizer = model.get_tokenizer()
    stop_token_ids = tokenizer.eos_token_id
    if not isinstance(stop_token_ids, List):
        stop_token_ids = [stop_token_ids]
    print('Using those stop tokens id: ', stop_token_ids)
        
    for idx in range(num_sequences):
        # Randomly select parameters within a desired range
        if idx == 0:
            temperature = 0.6
        else:
            temperature = random.uniform(0.6, 0.9)
        top_p = random.uniform(0.8, 0.95)
        top_k = random.choice([40, 50, 60])

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=2048,
            seed=random.randint(0, 2**32 - 1),
            #stop=["</s>"],
            stop_token_ids = stop_token_ids,
            presence_penalty=1.0,
            frequency_penalty=1.2,
            n=1,
        )
        modified_prompt = prompts  # or apply a function that randomly perturbs the prompt
        response = model.generate(modified_prompt, sampling_params=sampling_params)
        all_responses.append([output.outputs[0].text for output in response])
        all_temperature.append([temperature for output in response])
        all_top_p.append([top_p for output in response])
        all_top_k.append([top_k for output in response])

    return all_responses, all_temperature, all_top_p, all_top_k


def main():
    args = parse_arguments()

    # we use a contextual path only because we deploy the script via skypilot
    print("Reading parquet")
    df = pd.read_parquet(args.dataset)
    #df = df.sample(n=args.num_prompts, random_state=42)
    # Extract the specific column
    prompts = df["instruction"]

    # Initialize the LLM with your chosen model
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        enable_chunked_prefill=False,
    )
    # Generate multiple responses per prompt
    print("Generating responses")
    responses, temperatures, top_p, top_k = generate_responses(llm, prompts, num_sequences=args.num_sequences)

    # Create output dataframe with multiple response columns
    output_data = {"instruction": prompts}
    for i in range(args.num_sequences):
        output_data[f"response_{i+1}"] = responses[i]
        output_data[f"temperature_{i+1}"] = temperatures[i]
        output_data[f"top_p_{i+1}"] = top_p[i]
        output_data[f"top_k_{i+1}"] = top_k[i]

    # Create output dataframe and save
    df_output = pd.DataFrame(output_data)
    os.makedirs(f"{args.output_path}", exist_ok=True)
    output_file = os.path.join(args.output_path, "public_generated.parquet")
    df_output.to_parquet(output_file)


if __name__ == "__main__":
    main()
