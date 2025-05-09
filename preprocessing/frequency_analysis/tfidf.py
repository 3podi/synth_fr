import pandas as pd
import spacy
import argparse
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path
from time import time

# Load spaCy once
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

def count_lines_in_csv(file_path):
    """Count the number of lines in a CSV file"""
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return sum(1 for row in reader) - 1

def custom_tokenizer(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

def read_and_preprocess(file_path, column, total_lines):
    texts = []
    chunksize = 1000
    with tqdm(total=total_lines, desc="Reading & preprocessing") as pbar:
        for chunk in pd.read_csv(file_path, chunksize=chunksize, usecols=[column]):
            for text in chunk[column].dropna():
                texts.append(text)
                pbar.update(1)
    return texts

def compute_tfidf(texts, ngram_range=(1,1), max_features=None):
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        lowercase=True,
        stop_words=None,
        ngram_range=ngram_range,
        max_features=max_features,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix

def main(file_path, column, n):
    total_lines = count_lines_in_csv(file_path)

    t1= time()
    texts = read_and_preprocess(file_path, column, total_lines)

    print("Computing TF-IDF...")
    vectorizer, tfidf_matrix = compute_tfidf(texts, ngram_range=(1, n))
    t2 = time()
    print('Elapsed time: ', t2-t1)

    # Save the results
    output_dir = Path("preprocessing/frequency_analysis/results_frequency")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"tfidf_vectorizer_{n}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(output_dir / f"tfidf_matrix_{n}.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    print(f"TF-IDF complete: {tfidf_matrix.shape[0]} docs, {tfidf_matrix.shape[1]} features")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF with lemmatization and progress bar")
    parser.add_argument("file_path", type=str, help="CSV file path")
    parser.add_argument("column", type=str, help="Name of the text column")
    parser.add_argument("n", type=int, help="n-gram size (e.g., 1=unigram, 2=bigram)")

    args = parser.parse_args()
    main(args.file_path, args.column, args.n)
