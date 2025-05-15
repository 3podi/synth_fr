import os
import pandas as pd
import argparse
import unicodedata
import re
import string

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

def remove_accents(text):
    """Remove accents and special characters from Unicode text."""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

def normalize_text(text):
    # Lowercase
    text = text.lower()

    # Normalize common Unicode dashes to hyphen
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')   

    # Replace hyphens with spaces
    text = text.replace('-', ' ')
    
    # Remove accents
    text = remove_accents(text)
    
    # Remove invisible/non-printable characters
    text = ''.join(c for c in text if c.isprintable())
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # All punctuation except '-'
    punctuation_to_remove = string.punctuation.replace('-', '')
    # Remove all punctuation except '-'
    text = text.translate(str.maketrans('', '', punctuation_to_remove))
    
    return re.sub(r'\s+', ' ', text).strip()

if __name__ == "__main__":
    # Example usage
    #input_path = 'path/to/your/huge.csv'
    #output_dir = 'path/to/output/chunks'
    #split_csv(input_path, output_dir, chunk_size=100000)

    parser = argparse.ArgumentParser(description= 'Setting input and output path')
    parser.add_argument('big_boy_path', type=str, help='Path to the input big .csv file')
    parser.add_argument('output_dir', type=str, help='Path to the folder for saving output')
    parser.add_argument('--chunksize', type=int, default=10000, help='Number of rows in each sub file')

    args = parser.parse_args()
    split_csv(input_path=args.big_boy_path, output_dir=args.output_dir, chunk_size=args.chunksize)

