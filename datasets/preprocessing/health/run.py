import os
import random
from typing import List

import pandas as pd
from tqdm import tqdm
import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft_path", type=str, required=True, help="Path to the anonymous sft data to be splitted in public seed for sft and public seed for generation/scoring"
    )
    
    parser.add_argument(
        "--output_path", type=str, required=False,
        default="datasets/health/", help="Base path to save the data"
    )
    
    parser.add_argument(
        "--sft_size", type=int, required=True, help="Number of sft dataset size"
    )
    
    parser.add_argument(
        "--model_name", type=str, required=True, help="Chosen model"
    )

    return parser.parse_args()


def generate_public_and_private_seeds(input_path: str, output_dir: str, n1: int, seed: int = 42):
    """
    Samples N1 rows from a Parquet file and saves them as two separate files
    (public_seed.parquet and private_seed.parquet) in the output directory.

    Parameters:
        input_path (str): Path to the input .parquet file.
        output_dir (str): Directory to save the sampled files.
        n1 (int): Number of samples for the public seed.
        seed (int): Random seed for reproducibility.
    """
    df = pd.read_parquet(input_path)

    # Shuffle the full dataset reproducibly
    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    public_seed = shuffled_df.iloc[:n1]
    private_seed = shuffled_df.iloc[n1:]

    os.makedirs(output_dir, exist_ok=True)

    public_path = os.path.join(output_dir, "public_seed.parquet")
    private_path = os.path.join(output_dir, "private_seed.parquet")
    public_seed.to_parquet(public_path, index=False)
    private_seed.to_parquet(private_path, index=False)

    print(f"Saved {n1} public samples to: {public_path}")


if __name__ == "__main__":
    
    args = parse_arguments()
    
    output_dir = f"{args.output_path}/model={args.model_name.replace('/', '-')}_size={args.sft_size}"
    os.makedirs(output_dir, exist_ok=True)

    generate_public_and_private_seeds(
        input_path=args.sft_path,
        output_dir=output_dir,
        n1=args.sft_size
    )
    

    
    
