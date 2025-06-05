import numpy as np
import argparse
from scipy.sparse import csr_matrix, dok_matrix, issparse
import sys
import os 
import csv
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import pickle

# Add the current working directory (project root) to sys.path
sys.path.insert(0, os.getcwd())
from preprocessing.utils_preprocessing.utils import get_percentile_vocab, remove_symbols, normalize_text

def count_lines_in_csv(file_path):
    """Count the number of lines in a CSV file efficiently"""
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return sum(1 for row in reader) - 1

def NaiveBayesCSV(
    csv_path,
    text_column,
    label_column,
    input_vocab=None,
    batch_size=10000,
    dtype=np.float32,
    alpha=1.0
):
    V = len(input_vocab)
    word2idx = {w: i for i, w in enumerate(input_vocab)}
    
    # First pass: gather class names and total doc count
    class_set = set()
    total_docs = 0

    total_lines = count_lines_in_csv(csv_path)
    with tqdm(total=total_lines, desc="Processing chunks 1") as pbar:
        for chunk in pd.read_csv(csv_path, usecols=[text_column, label_column], chunksize=batch_size):
            chunk = chunk.dropna(subset=[text_column, label_column])
            labels = chunk[label_column].apply(lambda x: [remove_symbols(code) for code in str(x).strip().split()])

            for lbls in labels:
                if isinstance(lbls, list):
                    class_set.update(lbls)
                else:
                    class_set.add(lbls)
            total_docs += len(chunk)
            pbar.update(len(chunk))


    all_class_names = sorted(class_set)
    class2idx = {name: i for i, name in enumerate(all_class_names)}
    idx2class = {i: name for name, i in class2idx.items()}
    C = len(all_class_names)

    # Initialize counts
    class_doc_counts = np.zeros(C, dtype=dtype)
    word_counts = dok_matrix((C, V), dtype=dtype)

    # Second pass: build word-class counts
    with tqdm(total=total_lines, desc="Processing chunks 2") as pbar:
        for chunk in pd.read_csv(csv_path, usecols=[text_column, label_column], chunksize=batch_size):
            chunk = chunk.dropna(subset=[text_column, label_column])
            documents = chunk[text_column].apply(lambda x: normalize_text(str(x).replace('\n', ' '))).tolist()
            labels = chunk[label_column].apply(lambda x: [remove_symbols(code) for code in str(x).strip().split()])

            for doc, doc_labels in zip(documents, labels):
                if not isinstance(doc_labels, list):
                    doc_labels = [doc_labels]
                counts = defaultdict(int)
                for word in normalize_text(doc).split():
                    if word in word2idx:
                        counts[word2idx[word]] += 1
                for class_name in doc_labels:
                    c = class2idx[class_name]
                    class_doc_counts[c] += 1
                    for j, v in counts.items():
                        word_counts[c, j] += v
            
            pbar.update(len(chunk))


    # Convert to CSR for efficiency
    word_counts = word_counts.tocsr()

    # Apply smoothing and compute P(w|c)
    row_sums = word_counts.sum(axis=1).A1  # shape (C,)
    P_w_given_c = word_counts.copy()
    for c in range(C):
        P_w_given_c.data[P_w_given_c.indptr[c]:P_w_given_c.indptr[c+1]] += alpha
    row_sums += alpha * V
    P_w_given_c = P_w_given_c.multiply(1 / row_sums[:, None])

    # Compute P(c)
    P_c = class_doc_counts / class_doc_counts.sum()

    return P_w_given_c, P_c, idx2class

def show_top_words_per_class(P_w_given_c, vocab, top_k=10, class_names=None):
    """
    Display top_k words for each class based on P(w|c), supporting sparse matrices.
    
    Args:
        P_w_given_c: shape (C, V) — can be dense (ndarray) or sparse (csr_matrix)
        vocab: list or array of words
        top_k: number of words to show per class
        class_names: list of class names (optional)
    """
    C, V = P_w_given_c.shape
    vocab = np.array(list(vocab))

    for c in range(C):
        name = f"Class {c}" if class_names is None else class_names[c]
        print(f"\n📚 Top {top_k} words for {name}:")

        if issparse(P_w_given_c):
            row = P_w_given_c.getrow(c).toarray().ravel()  # shape (V,)
        else:
            row = P_w_given_c[c]

        top_indices = np.argsort(row)[::-1][:top_k]
        for rank, idx in enumerate(top_indices):
            prob = row[idx]
            print(f"{rank+1:>2}. {vocab[idx]:<15} (P={prob:.6f})")

def get_mutually_exclusive_top_words(P_w_given_c, vocab, top_k=10, class_names=None, search_k=None):
    """
    Finds top_k words for each class such that no word is shared between classes.
    
    Args:
        P_w_given_c: ndarray or sparse (C, V)
        vocab: list or array of vocabulary words
        top_k: number of unique words to assign per class
        class_names: optional list of class names
        search_k: how many top candidates to search from per class (default: 5 * top_k)
    """
    C, V = P_w_given_c.shape
    vocab = np.asarray(list(vocab))
    used_words = set()
    assigned_words = [[] for _ in range(C)]
    search_k = search_k or top_k * 5

    # Precompute sorted indices for all classes
    if issparse(P_w_given_c):
        sorted_indices = [np.argsort(P_w_given_c.getrow(c).toarray().ravel())[::-1] for c in range(C)]
    else:
        sorted_indices = [np.argsort(P_w_given_c[c])[::-1] for c in range(C)]

    for c in range(C):
        name = f"Class {c}" if class_names is None else class_names[c]
        print(f"\n📚 Top {top_k} exclusive words for {name}:")
        row = P_w_given_c.getrow(c).toarray().ravel() if issparse(P_w_given_c) else P_w_given_c[c]
        selected = []
        
        for idx in sorted_indices[c][:search_k]:
            word = vocab[idx]
            if word not in used_words:
                selected.append((word, row[idx]))
                used_words.add(word)
            if len(selected) == top_k:
                break
        
        #assigned_words[c] = selected
        for i, (word, prob) in enumerate(selected):
            print(f"{i+1:>2}. {word:<15} (P={prob:.6f})")
        
        if len(selected) < top_k:
            print(f"   ⚠ Only found {len(selected)} exclusive words.")

    return assigned_words

def get_mutually_exclusive_top_words2(P_w_given_c, vocab, top_k=10, class_names=None, search_k=None):
    """
    Memory-efficient version: select top_k mutually exclusive words for each class.
    
    Args:
        P_w_given_c: array (C, V) or sparse matrix
        vocab: list or array of words
        top_k: number of words to select per class
        class_names: optional list of class names
        search_k: how many top candidates to search from per class (default: 5 * top_k)
    """
    C, V = P_w_given_c.shape
    vocab = np.asarray(list(vocab))
    used_word_indices = set()
    search_k = search_k or top_k * 5

    for c in range(C):
        name = f"Class {c}" if class_names is None else class_names[c]
        print(f"\n📚 Top {top_k} exclusive words for {name}:")

        # Get row `c` as a dense 1D array if sparse
        row = P_w_given_c.getrow(c).toarray().ravel() if issparse(P_w_given_c) else P_w_given_c[c]

        # Get top `search_k` indices efficiently
        candidate_indices = np.argpartition(-row, search_k)[:search_k]
        candidate_indices = candidate_indices[np.argsort(-row[candidate_indices])]

        selected = []
        for idx in candidate_indices:
            if idx not in used_word_indices:
                selected.append((idx, row[idx]))
                used_word_indices.add(idx)
            if len(selected) == top_k:
                break

        for i, (idx, prob) in enumerate(selected):
            print(f"{i+1:>2}. {vocab[idx]:<15} (P={prob:.6f})")
        
        if len(selected) < top_k:
            print(f"   ⚠ Only found {len(selected)} exclusive words.")

def main(note_path, output_path, vocab_path, save_flag=False, col='input', col_codes='labels',chunk_size=500):

    # Limit vocab, by default to words in 25-75% percentile
    vocab = get_percentile_vocab(vocab_path, 85, 99.5)    
        
    P_w_given_c, P_c, idx2class = NaiveBayesCSV(csv_path=note_path,
                                                text_column=col,
                                                label_column=col_codes,
                                                input_vocab=vocab,
                                                batch_size=chunk_size)
    print('Finished')
    
    #get_mutually_exclusive_top_words2(P_w_given_c=P_w_given_c, class_names=idx2class, vocab=vocab, top_k=10)
    #show_top_words_per_class(P_w_given_c=P_w_given_c, vocab=vocab)

    if save_flag:
        
        os.makedirs(output_path, exist_ok=True)  # Ensure the directory exists

        # Save conditional probabilities
        np.save(f'{output_path}/naive_bayes_all_Pwc_output.npy', P_w_given_c)

        # Save class priors
        np.save(f'{output_path}/naive_bayes_all_Pc_output.npy', P_c)

        # Save vocabulary
        with open(f'{output_path}/naive_bayes_all_vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)

        # Save index-to-class mapping
        with open(f'{output_path}/naive_bayes_all_idx2class.pkl', 'wb') as f:
            pickle.dump(idx2class, f)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description= 'Setting notes, vocab and output path')
    parser.add_argument('note_path', type=str, help='Path to the notes')
    parser.add_argument('output_file_path', type=str, help='Path to folder where to store the results')
    parser.add_argument('--vocab_path', type=str, default=None, help='Path to 1-gram counter')
    parser.add_argument('--save', action='store_true', help='Optionally save P(w|c)')
    parser.add_argument('--column_text', type=str, default='input', help='Column name in the csv for text')
    parser.add_argument('--column_codes', type=str, default='labels', help='Column name in the csv for codes')
    parser.add_argument('--chunk_size', type=int, default=500, help='size of the chunks')

    args = parser.parse_args()
    
    note_path = args.note_path
    output_path = args.output_file_path
    vocab_path = args.vocab_path
    save_flag = args.save
    column_text = args.column_text
    column_codes = args.column_codes
    chunk_size = args.chunk_size
    
    main(note_path=note_path,
         output_path=output_path,
         vocab_path=vocab_path,
         save_flag=save_flag,
         col=column_text,
         col_codes=column_codes,
         chunk_size=chunk_size)
