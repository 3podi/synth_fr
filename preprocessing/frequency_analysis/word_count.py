import pandas as pd
import csv
from collections import Counter
import spacy
import argparse
from tqdm import tqdm
import pickle
from pathlib import Path

def process_large_csv(file_path, column):
    for chunk in pd.read_csv(file_path, chunksize=1, usecols=[column]):
        for value in chunk[column]:
            if pd.notnull(value):
                yield value

def generate_ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])

def count_lines_in_csv(file_path):
    """Count the number of lines in a CSV file efficiently"""
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return sum(1 for row in reader)

def WordCount(file,column,n):

    counter = Counter()
    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

    # Count total lines for a nice progress bar (can slow things down)
    total_lines = count_lines_in_csv(file)  # minus header
    chunksize = 1000

    with tqdm(total=total_lines, desc="Processing rows") as pbar:
        for chunk in pd.read_csv(file, chunksize=chunksize, usecols=[column]):
            for text in chunk[column].dropna():
                doc = nlp(text.lower())
                lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
                ngrams = generate_ngrams(lemmas, n)
                counter.update(ngrams)
                pbar.update(1)

    return counter


def main(file_path,column,n):

    counter = WordCount(file_path,column,n)

    print("Top 10 n-grams:")
    for k, v in counter.most_common(10):
        print(" ".join(k), ":", v)
        
    # Save
    output_dir = Path("preprocessing/frequency_analysis/results_frequency")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"ngrams_{n}_counter.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(counter, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description= 'Setting paths and n-grams')
    parser.add_argument('file_path', type=str, help='Path to the text file')
    parser.add_argument('column', type=str, help='Name of the text column')
    parser.add_argument('n', type=int, help='Integer that define what n-gram to count')

    args=parser.parse_args()

    main(args.file_path, args.column, args.n)