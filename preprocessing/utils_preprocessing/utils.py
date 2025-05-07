import os
import pandas as pd

def split_csv(input_path, output_dir, chunk_size=10000):
    """
    Splits a large CSV into smaller chunks.
    
    Args:
        input_path (str): Path to the input CSV file.
        output_dir (str): Directory where the output chunks will be saved.
        chunk_size (int): Number of rows per chunk.
    """
    os.makedirs(output_dir, exist_ok=True)

    reader = pd.read_csv(input_path, chunksize=chunk_size)
    for i, chunk in enumerate(reader):
        out_path = os.path.join(output_dir, f'chunk_{i:05}.csv')
        chunk.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    # Example usage
    #input_path = 'path/to/your/huge.csv'
    #output_dir = 'path/to/output/chunks'
    #split_csv(input_path, output_dir, chunk_size=100000)
