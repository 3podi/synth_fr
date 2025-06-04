import math
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

import argparse
import os
from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer
from tqdm import tqdm

from collections import Counter
import re
import unicodedata
import string
import pickle

french_stopwords = set([
    "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon", "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme", "comment", "dans", "des", "du", "dedans", "dehors", "depuis", "devrait", "doit", "donc", "dos", "droite", "début", "elle", "elles", "en", "encore", "essai", "est", "et", "eu", "fait", "faites", "fois", "font", "force", "haut", "hors", "ici", "il", "ils", "je", "juste", "la", "le", "les", "leur", "là", "ma", "maintenant", "mais", "mes", "mine", "moins", "mon", "mot", "même", "ni", "nommés", "notre", "nous", "nouveaux", "ou", "où", "par", "parce", "parole", "pas", "personnes", "peut", "peu", "pièce", "plupart", "pour", "pourquoi", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa", "sans", "ses", "seulement", "si", "sien", "son", "sont", "sous", "soyez", "sujet", "sur", "ta", "tandis", "tellement", "tels", "tes", "ton", "tous", "tout", "trop", "très", "tu", "voient", "vont", "votre", "vous", "vu", "ça", "étaient", "état", "étions", "été", "être", "mmol","mol","nmol"
])

def remove_accents(text):
    """Remove accents and special characters from Unicode text."""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

def normalize_text(text):

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

    # Replace all punctuation with whitespace
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # Lowercase
    text = text.lower()
    
    return re.sub(r'\s+', ' ', text).strip()

def get_percentile_vocab(vocab_path, lower_percentile=85, upper_percentile=99.5):
    """
    Get vocabulary of words in the specified percentile range of occurrence distribution.

    Args:
        vocab_path (str): Path to the pickle file containing word counts
        lower_percentile (int): Lower bound percentile (default: 85)
        upper_percentile (int): Upper bound percentile (default: 99.5)

    Returns:
        set: Vocabulary of words in the specified percentile range
    """
    # Load word counts
    print('Reading vocab')
    with open(vocab_path, 'rb') as f:
        word_counts = pickle.load(f)
    
    # Get all counts and sort them
    if isinstance(word_counts, Counter):
        counts = [count for word, count in word_counts.most_common()]
    else:
        counts = list(word_counts.values())
        
    # Calculate percentiles
    lower_threshold = np.percentile(counts, lower_percentile)
    upper_threshold = np.percentile(counts, upper_percentile)

    print(f'Lower and upper bound voca: {lower_threshold} - {upper_threshold}')
    
    stops = {normalize_text(stop) for stop in french_stopwords}
    
    # Filter words within the percentile range
    percentile_vocab = {
        normalize_text(word) for word, count in word_counts.items()
        if lower_threshold <= count <= upper_threshold
        if normalize_text(word) not in stops and len(normalize_text(word)) > 2
    }
            
    print('Len vocab: ', len(percentile_vocab))
    return percentile_vocab

def entropy(probs):
    """Compute entropy of a probability distribution."""
    return -sum(p * math.log(p + 1e-12) for p in probs)  # Add epsilon to avoid log(0)

def initialize_beam(tokenizer):
        
    vocab = get_percentile_vocab('/export/home/cse170020/Riccardo_T/word_counts/counter_full_vocab_lemmas_False.pkl', lower_percentile=97)
    
    return [(tokenizer.encode(normalize_text(w)),0) for w in vocab]

def initialize_beam_no_subwords(tokenizer):
    
    vocab = tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    
    print(len(inv_vocab))

    # Filter to get only full-word tokens (not subwords)
    full_word_token_ids = [
        token_id for token_id, token_str in inv_vocab.items()
        if token_str.startswith('▁')  # SentencePiece: full word tokens
    ]

    print(len(full_word_token_ids))
    
    # Initialize the beam with full-word tokens
    return [([token_id], 0.0) for token_id in full_word_token_ids]

def to_token_prob_topk(infinigram_dict, top_k=10):
    token_results = infinigram_dict['result_by_token_id']
    sorted_probs = sorted(
        ((int(k), v['prob']) for k, v in token_results.items()),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_probs[:top_k]
    #return [(k, v['prob']) for k, v in token_results.items()]

def analyze_entropy(engine, tokenizer, beam_size=10, max_len=10, num_seqs=10):
    entropy_by_step = defaultdict(list)
    #beam = [([], 0)]  # Start with empty prefix
    #beam = initialize_beam_no_subwords(tokenizer)
    beam = initialize_beam(tokenizer)
    #beam = beam[:num_seqs]

    for step in range(1, max_len + 1):
        print(f"\nStep {step}")
        new_beam = []

        for seq, log_p in beam:
            top_next = to_token_prob_topk(engine.ntd(prompt_ids=seq), beam_size)
            
            if len(top_next)==0:
                continue

            probs = [prob for _, prob in top_next]
            H = entropy(probs)
            entropy_by_step[step].append(H)

            print(f"Seq: {[tokenizer.convert_ids_to_tokens(seq)]}, Entropy: {H:.4f}")

            for token, prob in top_next:
                if prob <= 0.0:
                    continue
                new_seq = seq + [token]
                new_logp = log_p + math.log(prob)
                new_beam.append((new_seq, new_logp))

        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:num_seqs]

    return entropy_by_step

def plot_entropy_with_variance(entropy_by_step, save_path='../entropy_plot_from_vocab.png'):

    steps = sorted(entropy_by_step.keys())
    avg_entropies = [np.mean(entropy_by_step[s]) for s in steps]
    std_entropies = [np.std(entropy_by_step[s]) for s in steps]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, avg_entropies, marker='o', label='Mean Entropy')
    plt.fill_between(
        steps,
        [m - std for m, std in zip(avg_entropies, std_entropies)],
        [m + std for m, std in zip(avg_entropies, std_entropies)],
        color='blue',
        alpha=0.2,
        label='±1 Std Dev'
    )
    plt.title("Entropy per Step with Variance")
    plt.xlabel("Generation Step")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid(True)

    # Save the plot to file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

    plt.close()  # Close the figure to free memory

def main(index_folder, output_folder, n, top_k, tokenizer):
    
    #tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False, add_bos_token=False, add_eos_token=False)
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=os.environ.get('HF_TOKEN'), use_fast=False, add_bos_token=False, add_eos_token=False)

    engine = InfiniGramEngine(index_dir=index_folder, eos_token_id=tokenizer.eos_token_id)
    
    entropy_by_step = analyze_entropy(engine, tokenizer, max_len=n, beam_size=top_k, num_seqs=top_k)
    
    plot_entropy_with_variance(entropy_by_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beam search to find the most common n-grams.")
    parser.add_argument("index_folder", help="Path to folder with the infinigram indexing")
    parser.add_argument("output_folder", help="Path to output n-grams")
    parser.add_argument("n", type=int, help="N-grams size")
    parser.add_argument("top_k", type=int, help="Top k results to keep for each beam")
    parser.add_argument("--tokenizer", type=str, default='gpt2', help="Tokenizer to use")

    args = parser.parse_args()

    main(args.index_folder, args.output_folder, args.n, args.top_k, args.tokenizer)