import pandas as pd
from collections import Counter
import spacy
import argparse
from tqdm import tqdm
import pickle


def process_large_csv(file_path, column):
    for chunk in pd.read_csv(file_path, chunksize=1, usecols=[column]):
        for value in chunk[column]:
            if pd.notnull(value):
                yield value

def generate_ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])

def count_lines(file):
    # Count lines excluding header
    with open(file, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1

def WordCount(file,column,n):

    counter = Counter()
    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

    # Count total lines for a nice progress bar (can slow things down)
    total_lines = sum(1 for _ in open(file)) - 1  # minus header
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

    print(counter)

    # Save
    with open(f"preprocessing/frequency_analysis/results_frequency/ngrams_{n}_counter.pkl", "wb") as f:
        pickle.dump(counter, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description= 'Setting paths and n-grams')
    parser.add_argument('file_path', type=str, help='Path to the text file')
    parser.add_argument('column', type=str, help='Name of the text column')
    parser.add_argument('n', type=int, help='Integer that define what n-gram to count')

    args=parser.parse_args()

    main(args.file_path, args.column, args.n)