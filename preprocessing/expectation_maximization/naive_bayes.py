import numpy as np
import argparse
from scipy.sparse import csr_matrix, dok_matrix, issparse
import sys
import os 

# Add the current working directory (project root) to sys.path
sys.path.insert(0, os.getcwd())
from preprocessing.utils_preprocessing.utils import get_notes, get_percentile_vocab

def NaiveBayes(documents, labels, input_vocab=None, batch_size=500, dtype=np.float32, alpha=1.0):
    V = len(input_vocab)
    word2idx = {w: i for i, w in enumerate(input_vocab)}
    D = len(documents)
    
    # Build sparse matrix X of shape (D, V)
    row, col, data = [], [], []
    for i, doc in enumerate(documents):
        counts = {}
        for word in doc.split(' '):
            if word in word2idx:
                idx = word2idx[word]
                counts[idx] = counts.get(idx, 0) + 1
        for j, count in counts.items():
            row.append(i)
            col.append(j)
            data.append(count)
    X = csr_matrix((data, (row, col)), shape=(D, V), dtype=dtype)

    # Extract unique class names and create mapping
    all_class_names = sorted(set(c for doc_labels in labels for c in doc_labels))

    class2idx = {name: i for i, name in enumerate(all_class_names)}
    #idx2class = {i: name for name, i in class2idx.items()}
    C = len(all_class_names)

    class_doc_counts = np.zeros(C, dtype=dtype)
    word_counts = dok_matrix((C, V), dtype=dtype)

    for start in range(0, D, batch_size):
        end = min(start + batch_size, D)
        X_batch = X[start:end]
        labels_batch = labels[start:end]
        
        for i, doc_labels in enumerate(labels_batch):
            row = X_batch[i]
            idxs = row.indices
            vals = row.data

            for class_name in doc_labels:
                c = class2idx[class_name]
                class_doc_counts[c] += 1
                for j, v in zip(idxs, vals):
                    word_counts[c, j] += v

    # Convert to CSR for efficient row-wise ops
    word_counts = word_counts.tocsr()

    # Compute smoothed probabilities
    row_sums = word_counts.sum(axis=1).A1  # shape (C,)
    P_w_given_c = word_counts.copy()
    for c in range(C):
        P_w_given_c.data[P_w_given_c.indptr[c]:P_w_given_c.indptr[c+1]] += alpha
    row_sums += alpha * V
    P_w_given_c = P_w_given_c.multiply(1 / row_sums[:, None])

    P_c = class_doc_counts / class_doc_counts.sum()
    return P_w_given_c, P_c

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

def main(note_path, output_path, vocab_path, save_flag=False):

    # Limit vocab, by default to words in 25-75% percentile
    vocab = get_percentile_vocab(vocab_path, 25, 50)    
    
    documents, labels = get_notes(note_path,labels=True)
    
    P_w_given_c, P_c = NaiveBayes(documents=documents, labels=labels, input_vocab=vocab)
    
    print('Finished')
    get_mutually_exclusive_top_words2(P_w_given_c=P_w_given_c, vocab=vocab, top_k=20)
    #show_top_words_per_class(P_w_given_c=P_w_given_c, vocab=vocab)

    if save_flag:
        np.save(f'{output_path}naive_bayes_output.npy', P_w_given_c)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description= 'Setting notes, vocab and output path')
    parser.add_argument('note_path', type=str, help='Path to the notes')
    parser.add_argument('output_file_path', type=str, help='Path to folder where to store the results')
    parser.add_argument('--vocab_path', type=str, default=None, help='Path to 1-gram counter')
    parser.add_argument('--save', action='store_true', help='Optionally save P(w|c)')
    
    args = parser.parse_args()
    
    note_path = args.note_path
    output_path = args.output_file_path
    vocab_path = args.vocab_path
    save_flag = args.save
    
    main(note_path=note_path, output_path=output_path, vocab_path=vocab_path, save_flag=save_flag)