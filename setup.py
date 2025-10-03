import os
import shutil
import argparse
from datasets import load_dataset

def setup_dataset_repo(model_name, private_seed_path, public_seed_path, dataset_size):
    """
    Creates a dataset folder structure:
    synth_fr/datasets/health/model=<model_name>_size=<dataset_size>/
    and copies the private/public seed files into it.
    """
    # Base path
    base_dir = os.path.join("datasets", "health")
    os.makedirs(base_dir, exist_ok=True)

    # Dataset folder name
    folder_name = f"model={model_name.replace('/', '-')}_size={dataset_size}"
    dataset_dir = os.path.join(base_dir, folder_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Copy seeds
    private_dest = os.path.join(dataset_dir, "private_seed.parquet")
    public_dest = os.path.join(dataset_dir, "public_seed.parquet")

    shutil.copy(private_seed_path, private_dest)
    shutil.copy(public_seed_path, public_dest)

    print(f"Dataset structure created at: {dataset_dir}")
    print(f"Private seed copied to: {private_dest}")
    print(f"Public seed copied to: {public_dest}")

def download_hf_datasets():
    """
    Downloads specific Hugging Face datasets into the current directory.
    """
    hf_datasets = ["3podi/coded_complete", "3podi/cim_synonymes"]
    
    for ds_name in hf_datasets:
        print(f"Downloading {ds_name}...")
        dataset = load_dataset(ds_name, cache_dir=os.getcwd())
        print(f"{ds_name} downloaded to current directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup dataset repo for synth_fr")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--private_seed", type=str, required=True, help="Path to private seed parquet file")
    parser.add_argument("--public_seed", type=str, required=True, help="Path to public seed parquet file")
    parser.add_argument("--dataset_size", type=int, required=True, help="Dataset size to include in folder name")
    parser.add_argument("--download_hf", action="store_false", help="Download HF files to current directory")

    args = parser.parse_args()

    setup_dataset_repo(args.model_name, args.private_seed, args.public_seed, args.dataset_size)

    if args.download_hf:
        download_hf_datasets()

