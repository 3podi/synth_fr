import os
import pandas as pd
import argparse
import unicodedata
import re
import string
import csv
import pickle
import numpy as np
import ast

french_stopwords = set([
    "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon", "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme", "comment", "dans", "des", "du", "dedans", "dehors", "depuis", "devrait", "doit", "donc", "dos", "droite", "début", "elle", "elles", "en", "encore", "essai", "est", "et", "eu", "fait", "faites", "fois", "font", "force", "haut", "hors", "ici", "il", "ils", "je", "juste", "la", "le", "les", "leur", "là", "ma", "maintenant", "mais", "mes", "mine", "moins", "mon", "mot", "même", "ni", "nommés", "notre", "nous", "nouveaux", "ou", "où", "par", "parce", "parole", "pas", "personnes", "peut", "peu", "pièce", "plupart", "pour", "pourquoi", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa", "sans", "ses", "seulement", "si", "sien", "son", "sont", "sous", "soyez", "sujet", "sur", "ta", "tandis", "tellement", "tels", "tes", "ton", "tous", "tout", "trop", "très", "tu", "voient", "vont", "votre", "vous", "vu", "ça", "étaient", "état", "étions", "été", "être"
])

import spacy
nlp = spacy.load("fr_core_news_sm")
french_stopwords = set(nlp.Defaults.stop_words)
del nlp

def split_csv(input_path, output_dir, chunk_size=10000):
    """
    Splits a large CSV into smaller chunks.
    
    Args:
        input_path (str): Path to the input CSV file.
        output_dir (str): Directory where the output chunks will be saved.
        chunk_size (int): Number of rows per chunk.
    """
    os.makedirs(output_dir, exist_ok=True)

    reader = pd.read_csv(input_path, chunksize=chunk_size)
    for i, chunk in enumerate(reader):
        out_path = os.path.join(output_dir, f'chunk_{i:05}.csv')
        chunk.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

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

def get_notes(file_path,column='input', labels=False):
    """
    Read a CSV file and return a list of all texts from the 'text' column.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        list: List of text strings from the 'text' column
    """
    texts = []
    codes = []

    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # Check if 'text' column exists
        if column not in reader.fieldnames:
            raise ValueError(f"CSV file does not contain a {column} column")
        
        print('Reading documents..')
        for row in reader:
            texts.append(normalize_text(row[column].replace('\n', ' ')))
            if labels:
                predicted_expressions = ast.literal_eval(row['labels'])
                if not isinstance(predicted_expressions, list):
                    predicted_expressions = []
                codes.append(predicted_expressions)


    if labels:
        return texts, codes
    else:
        return texts

def get_percentile_vocab(vocab_path, lower_percentile=50, upper_percentile=80):
    """
    Get vocabulary of words in the specified percentile range of occurrence distribution.

    Args:
        vocab_path (str): Path to the pickle file containing word counts
        lower_percentile (int): Lower bound percentile (default: 25)
        upper_percentile (int): Upper bound percentile (default: 75)

    Returns:
        set: Vocabulary of words in the specified percentile range
    """
    # Load word counts
    print('Reading vocab')
    with open(vocab_path, 'rb') as f:
        word_counts = pickle.load(f)
    
    # Get all counts and sort them
    counts = [count for word, count in word_counts.most_common()]

    # Calculate percentiles
    lower_threshold = np.percentile(counts, lower_percentile)
    upper_threshold = np.percentile(counts, upper_percentile)
    
    stops = {normalize_text(stop) for stop in french_stopwords}
    
    # Filter words within the percentile range
    percentile_vocab = {
        normalize_text(word) for word, count in word_counts.items()
        if lower_threshold <= count <= upper_threshold
        if normalize_text(word) not in stops and len(normalize_text(word)) > 2
    }

    return percentile_vocab

if __name__ == "__main__":
    # Example usage
    #input_path = 'path/to/your/huge.csv'
    #output_dir = 'path/to/output/chunks'
    #split_csv(input_path, output_dir, chunk_size=100000)

    parser = argparse.ArgumentParser(description= 'Setting input and output path')
    parser.add_argument('big_boy_path', type=str, help='Path to the input big .csv file')
    parser.add_argument('output_dir', type=str, help='Path to the folder for saving output')
    parser.add_argument('--chunksize', type=int, default=10000, help='Number of rows in each sub file')

    args = parser.parse_args()
    split_csv(input_path=args.big_boy_path, output_dir=args.output_dir, chunk_size=args.chunksize)

