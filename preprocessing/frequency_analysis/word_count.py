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

def WordCount2(file, column, n, vocab_size=10000, vocab_path=None, useLemma=False):
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
    
    print('vocab_path: ', vocab_path)

    if vocab_path is None:
        print(f'No vocab_path given; computing the words count.')
        # First pass: Compute word frequencies to limit vocabulary
        word_counts = Counter()
        with tqdm(total=total_lines, desc="Computing word frequencies") as pbar:
            for chunk in pd.read_csv(file, chunksize=chunksize, usecols=[column]):
                for text in chunk[column].dropna():
                    doc = nlp(text.lower())
                    if useLemma:
                        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
                    else:
                        tokens = [token for token in doc if token.is_alpha and not token.is_stop]
                    word_counts.update(tokens)
                    pbar.update(1)

            print('Number of words in data: ', len(word_counts))
            with open(f'../counter_1gram_lemmas_{useLemma}.pkl', 'wb') as f:
                pickle.dump(word_counts, f)
            
            return True
    
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
                if useLemma:
                    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
                else:
                    tokens = [token for token in doc if token.is_alpha and not token.is_stop]
                ngrams = generate_ngrams(tokens, n)
                ngram_counter.update(ngrams)
                pbar.update(1)

    return ngram_counter

def save_chunk_word_counts_to_csv(file, column, useLemma, chunksize, output_dir):
    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
    Path(output_dir).mkdir(exist_ok=True)
    total_lines = count_lines_in_csv(file)
    chunk_id = 0

    with tqdm(total=total_lines, desc="Counting words per chunk") as pbar:
        for chunk in pd.read_csv(file, chunksize=chunksize, usecols=[column]):
            chunk_counter = Counter()
            for text in chunk[column].dropna():
                doc = nlp(text.lower())
                tokens = (
                    token.lemma_ if useLemma else token.text
                    for token in doc
                    if token.is_alpha and not token.is_stop
                )
                chunk_counter.update(tokens)
                pbar.update(1)

            output_file = Path(output_dir) / f"chunk_{chunk_id}.csv"
            with open(output_file, "w", encoding="utf-8") as f:
                for word, count in chunk_counter.items():
                    f.write(f"{word},{count}\n")
            chunk_id += 1

def merge_word_counts_from_csvs(input_dir, output_path):
    input_dir = Path(input_dir)
    csv_files = sorted(input_dir.glob("chunk_*.csv"))

    df_list = []
    for file in csv_files:
        df = pd.read_csv(file, names=["word", "count"])
        df_list.append(df)

    all_counts = pd.concat(df_list)
    final_counts = all_counts.groupby("word", as_index=False).sum()
    final_counts.sort_values("count", ascending=False, inplace=True)
    final_counts.to_csv(output_path, index=False)
    return final_counts


def WordCount3(file, column, n, vocab_size=10000, vocab_path=None, useLemma=False):
    """
    Count n-grams in a CSV file with memory-safe chunked word counting.

    Args:
        file (str): Path to the CSV file.
        column (str): Name of the column containing text.
        n (int): The n in n-grams.
        vocab_size (int): Size of the vocabulary to limit to.
        vocab_path (str): Path to an existing vocab CSV, if available.
        useLemma (bool): Whether to use lemmas.
    """
    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
    total_lines = count_lines_in_csv(file)
    chunksize = 10000
    wordcount_dir = Path("tmp_word_counts")
    wordcount_dir.mkdir(exist_ok=True)

    # STEP 1: Word count if no vocab_path
    if vocab_path is None:
        print("No vocab_path given; computing chunked word counts.")
        save_chunk_word_counts_to_csv(file, column, useLemma, chunksize, wordcount_dir)
        vocab_df = merge_word_counts_from_csvs(wordcount_dir, "merged_vocab.csv")
        # Save full vocab as .pkl for later use if needed
        with open(f"../word_counts/counter_full_vocab_lemmas_{useLemma}.pkl", "wb") as f:
            pickle.dump(dict(zip(vocab_df["word"], vocab_df["count"])), f)
    else:
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        print(f"Loaded vocabulary with {len(vocab)} words.")

    # STEP 2: Compute n-grams
    ngram_counter = Counter()
    with tqdm(total=total_lines, desc="Processing rows for n-grams") as pbar:
        for chunk in pd.read_csv(file, chunksize=chunksize, usecols=[column]):
            for text in chunk[column].dropna():
                doc = nlp(text.lower())
                tokens = [
                    token.lemma_ if useLemma else token.text
                    for token in doc
                    if token.is_alpha and not token.is_stop and
                       (token.lemma_ if useLemma else token.text) in vocab
                ]
                ngrams = generate_ngrams(tokens, n)
                ngram_counter.update(ngrams)
                pbar.update(1)

    return ngram_counter


def main(file_path,column,n,vocab_path, save_flag, lemma_flag):

    counter = WordCount3(file_path,column,n,vocab_path=vocab_path,useLemma=lemma_flag)

    print("Top 10 n-grams:")
    for k, v in counter.most_common(10):
        print(" ".join(k), ":", v)
        
    # Save
    if save_flag:
        output_dir = Path("../word_counts")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"ngrams_{n}_counter_lemmas_{lemma_flag}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(counter, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description= 'Setting paths and n-grams')
    parser.add_argument('file_path', type=str, help='Path to the text file')
    parser.add_argument('column', type=str, help='Name of the text column')
    parser.add_argument('n', type=int, help='Integer that define what n-gram to count')
    parser.add_argument('--vocab_path', type=str, default=None, help='path to load vocab')
    parser.add_argument('--save', action='store_true', help='Optionally save n-gram count')
    parser.add_argument('--lemma', action='store_true', help='Optionally lemmatize words')

    args=parser.parse_args()
    
    print('args pqth: ', args.vocab_path)

    main(args.file_path, args.column, args.n, args.vocab_path, args.save, args.lemma)