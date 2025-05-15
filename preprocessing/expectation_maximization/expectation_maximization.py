import numpy as np
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from tqdm import tqdm
import pickle
from preprocessing.utils_preprocessing.utils import get_notes, get_percentile_vocab

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

def main(note_path, output_path, vocab_path, num_classes=10, vocab_size=10000):

    # Limit vocab, by default to words in 25-75% percentile
    vocab = get_percentile_vocab(vocab_path)    
    print('Number of words in my vocabulary: ', len(vocab))

    documents = get_notes(note_path)

    P_w_given_c = ExpectationMaximization(documents=documents,num_classes=num_classes,input_vocab=vocab)

    print(P_w_given_c)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description= 'Setting notes, vocab and output path and em params')
    parser.add_argument('note_path', type=str, help='Path to the notes')
    parser.add_argument('output_file_path', type=str, help='Path to folder where to store the results')
    parser.add_argument('--vocab_path', type=str, default=None, help='Path to dictionary for keywords extraction')
    parser.add_argument('--num_classes', type=int, default=10, help='List of similarity thresholds')
    
    args = parser.parse_args()
    
    note_path = args.note_path
    output_path = args.output_file_path
    vocab_path = args.vocab_path
    num_classes = args.num_classes

    main(note_path=note_path, output_path=output_path, vocab_path=vocab_path, num_classes=num_classes)