import numpy as np
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from scipy.sparse import csr_matrix
from preprocessing.utils_preprocessing.utils import get_notes, get_percentile_vocab

def show_top_words_per_class(P_w_given_c, vocab, top_k=50, class_names=None):
    """
    Display top_k words for each class based on P(w|c).
    
    Args:
        P_w_given_c: shape (C, V)
        vocab: list of words
        top_k: number of words to show per class
        class_names: list of class names (optional)
    """
    C, V = P_w_given_c.shape
    vocab = np.array(list(vocab))

    for c in range(C):
        name = f"Class {c}" if class_names is None else class_names[c]
        print(f"\n📚 Top {top_k} words for {name}:")
        top_indices = np.argsort(P_w_given_c[c])[::-1][:top_k]
        for rank, idx in enumerate(top_indices):
            prob = P_w_given_c[c, idx]
            print(f"{rank+1:>2}. {vocab[idx]:<15} (P={prob:.4f})")

def ExpectationMaximization(documents, num_classes, input_vocab=None):

    # Vectorizer restricted to input vocabulary
    vectorizer = CountVectorizer(vocabulary=input_vocab)
    X = vectorizer.fit_transform(documents).toarray() # [num_docs, vocab_size]
    vocab = vectorizer.get_feature_names_out()
    num_docs, vocab_size = X.shape

    # === Initialize ===
    P_c = np.full(num_classes, 1 / num_classes)  # P(c)
    P_w_given_c = np.random.dirichlet(np.ones(vocab_size), size=num_classes)  # P(w|c) [vocab_size, N]
    P_c_given_d = np.zeros((num_docs, num_classes))  # P(c | d) [num_docs, N]

    # === EM Loop ===
    for iteration in tqdm(range(20), desc='Iteration EM'):
        # E-step
        for d in range(num_docs):
            log_probs = []
            for c in range(num_classes):
                log_likelihood = np.sum(X[d] * np.log(P_w_given_c[c] + 1e-10))
                log_probs.append(np.log(P_c[c]) + log_likelihood)
            max_log = max(log_probs)
            probs = np.exp(np.array(log_probs) - max_log)
            P_c_given_d[d] = probs / np.sum(probs)

        # M-step
        P_c = P_c_given_d.mean(axis=0)
        for c in range(num_classes):
            weighted_counts = np.zeros(vocab_size)
            total_words = 0
            for d in range(num_docs):
                weighted_counts += X[d] * P_c_given_d[d, c]
                total_words += np.sum(X[d]) * P_c_given_d[d, c]
            P_w_given_c[c] = weighted_counts / (total_words + 1e-10)

    # === Show Top Words Per Class ===
    top_k = 5
    for c in range(num_classes):
        print(f"\nClass {c} (P(c)={P_c[c]:.2f}):")
        top_indices = np.argsort(P_w_given_c[c])[::-1][:top_k]
        for idx in top_indices:
            print(f"  {vocab[idx]:<10} -> P(w|c) = {P_w_given_c[c][idx]:.3f}")
    
    return P_w_given_c

def ExpectationMaximization2(documents, num_classes, input_vocab=None):
    
    # Vectorize using fixed vocabulary
    vectorizer = CountVectorizer(vocabulary=input_vocab)
    X = vectorizer.fit_transform(documents).toarray().astype(np.float32)  # shape (D, V)
    vocab = vectorizer.get_feature_names_out()
    D, V = X.shape

    # Initialize
    P_c = np.full((num_classes,), 1 / num_classes, dtype=np.float32)      # shape (C,)
    P_w_given_c = np.random.dirichlet(np.ones(V), size=num_classes).astype(np.float32)  # shape (C, V)

    for iteration in range(3):
        # ---------- E-step ----------
        # Compute log P(w|c) for all docs and classes
        # log_likelihoods: shape (D, C)
        log_P_w_given_c = np.log(P_w_given_c + 1e-10).astype(np.float32)   # shape (C, V)
        log_likelihoods = X @ log_P_w_given_c.T                            # shape (D, C)

        log_joint = log_likelihoods + np.log(P_c + 1e-10)[None, :]         # shape (D, C)
        max_log = np.max(log_joint, axis=1, keepdims=True)                # for numerical stability
        probs = np.exp(log_joint - max_log)                               # shape (D, C)
        P_c_given_d = probs / np.sum(probs, axis=1, keepdims=True)        # shape (D, C)

        # ---------- M-step ----------
        P_c = np.mean(P_c_given_d, axis=0)                                 # shape (C,)

        # Compute expected word counts: P_c_given_d.T @ X gives shape (C, V)
        weighted_counts = P_c_given_d.T @ X                                # shape (C, V)
        total_words_per_class = weighted_counts.sum(axis=1, keepdims=True)  # shape (C, 1)
        P_w_given_c = weighted_counts / (total_words_per_class + 1e-10)    # shape (C, V)

    # ---------- Output ----------
    top_k = 5
    for c in range(num_classes):
        print(f"\nClass {c} (P(c)={P_c[c]:.2f}):")
        top_indices = np.argsort(P_w_given_c[c])[::-1][:top_k]
        for idx in top_indices:
            print(f"  {vocab[idx]:<10} -> P(w|c) = {P_w_given_c[c][idx]:.3f}")

def ExpectationMaximization3(documents, num_classes, input_vocab=None, batch_size=1000, dtype=np.float32):
    """
    Run EM algorithm to find P(w | c) and P(c) using batching and sparse matrix.
    
    Args:
        docs: List of documents (each a list of words).
        class_probs: Initial P(c), shape (num_classes,)
        vocab: List of words to include.
        num_classes: Number of classes.
        max_iter: EM iterations.
        batch_size: Batch size for processing documents.
        dtype: numpy dtype (e.g., np.float32 or np.float16)
    
    Returns:
        P_w_given_c: ndarray (num_classes, vocab_size)
        P_c: ndarray (num_classes,)
    """
    V = len(input_vocab)
    C = num_classes
    word2idx = {w: i for i, w in enumerate(input_vocab)}
    D = len(documents)
    max_iter = 10

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

    # Initialize
    P_c = np.full((num_classes,), 1 / num_classes, dtype=dtype)      # shape (C,)
    P_w_given_c = np.random.rand(C, V).astype(dtype)
    P_w_given_c /= P_w_given_c.sum(axis=1, keepdims=True)

    for it in range(max_iter):
        expected_counts = np.zeros((C, V), dtype=dtype)
        class_totals = np.zeros(C, dtype=dtype)

        for start in tqdm(range(0, D, batch_size), desc=f"EM Iteration"):
            end = min(start + batch_size, D)
            X_batch = X[start:end]  # shape (B, V)
            B = X_batch.shape[0]

            log_P_w_given_c = np.log(P_w_given_c + 1e-9)
            log_likelihoods = X_batch @ log_P_w_given_c.T  # shape (B, C)
            log_likelihoods += np.log(P_c + 1e-9)

            max_log = log_likelihoods.max(axis=1).reshape(-1, 1)
            probs = np.exp(log_likelihoods - max_log)
            P_c_given_d = probs / probs.sum(axis=1, keepdims=True)  # shape (B, C)

            weighted = P_c_given_d.T @ X_batch.toarray()  # shape (C, V)
            expected_counts += weighted
            class_totals += P_c_given_d.sum(axis=0)

        # M-step
        P_w_given_c = expected_counts / expected_counts.sum(axis=1, keepdims=True)
        P_c = class_totals / D

    # ---------- Output ----------
    #top_k = 5
    #for c in range(num_classes):
    #    print(f"\nClass {c} (P(c)={P_c[c]:.2f}):")
    #    top_indices = np.argsort(P_w_given_c[c])[::-1][:top_k]
    #    for idx in top_indices:
    #        print(f"  {input_vocab[idx]:<10} -> P(w|c) = {P_w_given_c[c][idx]:.3f}")

    return P_w_given_c, P_c


def main(note_path, output_path, vocab_path, num_classes=10, vocab_size=10000):

    # Limit vocab, by default to words in 25-75% percentile
    vocab = get_percentile_vocab(vocab_path)    
    print('Number of words in my vocabulary: ', len(vocab))

    documents = get_notes(note_path)

    P_w_given_c, P_c = ExpectationMaximization3(documents=documents,num_classes=num_classes,input_vocab=vocab)

    show_top_words_per_class(P_w_given_c=P_w_given_c, vocab=vocab)

    print(P_w_given_c)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description= 'Setting notes, vocab and output path and em params')
    parser.add_argument('note_path', type=str, help='Path to the notes')
    parser.add_argument('output_file_path', type=str, help='Path to folder where to store the results')
    parser.add_argument('--vocab_path', type=str, default=None, help='Path to dictionary for keywords extraction')
    parser.add_argument('--num_classes', type=int, default=100, help='List of similarity thresholds')
    
    args = parser.parse_args()
    
    note_path = args.note_path
    output_path = args.output_file_path
    vocab_path = args.vocab_path
    num_classes = args.num_classes

    main(note_path=note_path, output_path=output_path, vocab_path=vocab_path, num_classes=num_classes)