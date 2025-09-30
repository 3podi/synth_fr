import os
import shutil
import argparse

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
    folder_name = f"model={model_name.replace("/","-")}_size={dataset_size}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup dataset repo for synth_fr")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--private_seed", type=str, required=True, help="Path to private seed parquet file")
    parser.add_argument("--public_seed", type=str, required=True, help="Path to public seed parquet file")
    parser.add_argument("--dataset_size", type=int, required=True, help="Dataset size to include in folder name")

    args = parser.parse_args()

    setup_dataset_repo(args.model_name, args.private_seed, args.public_seed, args.dataset_size)
