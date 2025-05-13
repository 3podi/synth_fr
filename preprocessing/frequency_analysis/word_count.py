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
        return sum(1 for row in reader) - 1

#def get_vocab(file,column,nlp,vocab_size=10000):
#    df = pd.read_csv(file)
#    all_text = ' '.join(df[column]).split()
#    word_counts = Counter(all_text)
#
#    print('Number of vocabs in the dataset: ', len(word_counts))
#    # Keep only the most frequent words
#    vocab = set(word for word, count in word_counts.most_common(vocab_size))
#    return vocab

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

def WordCount2(file, column, n, vocab_size=10000, vocab_path=None):
    """
    Count n-grams in a CSV file with vocabulary limitation.

    Args:
        file (str): Path to the CSV file.
        column (str): Name of the column containing text.
        n (int): The n in n-grams.
        vocab_size (int): Size of the vocabulary to limit to.
    """
    # Initialize spaCy model
    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

    # Count total lines for a nice progress bar (can slow things down)
    total_lines = count_lines_in_csv(file)
    chunksize = 1000

    if vocab_path is None:
        print(f'No vocab_path given; computing the words count.')
        # First pass: Compute word frequencies to limit vocabulary
        word_counts = Counter()
        with tqdm(total=total_lines, desc="Computing word frequencies") as pbar:
            for chunk in pd.read_csv(file, chunksize=chunksize, usecols=[column]):
                for text in chunk[column].dropna():
                    doc = nlp(text.lower())
                    lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
                    word_counts.update(lemmas)
                    pbar.update(1)

            print('Number of words in data: ', len(word_counts))
            with open('../counter_1gram.pkl', 'wb') as f:
                pickle.dump(word_counts, f)
    
    else:
        with open(vocab_path, 'rb') as f:
            word_counts = pickle.load(f)

    # Limit vocabulary to the most frequent words
    vocab = set(word for word, count in word_counts.most_common(vocab_size))
    del word_counts
    print('Number of words in my vocabulary: ', len(vocab))

    # Second pass: Compute n-grams with limited vocabulary
    ngram_counter = Counter()
    with tqdm(total=total_lines, desc="Processing rows") as pbar:
        for chunk in pd.read_csv(file, chunksize=chunksize, usecols=[column]):
            for text in chunk[column].dropna():
                doc = nlp(text.lower())
                lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop and token.lemma_ in vocab]
                ngrams = generate_ngrams(lemmas, n)
                ngram_counter.update(ngrams)
                pbar.update(1)

    return ngram_counter


def main(file_path,column,n,vocab_path):

    counter = WordCount2(file_path,column,n,vocab_path)

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
    parser.add_argument('--vocab_path', type=str, default=None, help='path to load vocab')

    args=parser.parse_args()

    main(args.file_path, args.column, args.n, args.vocab_path)