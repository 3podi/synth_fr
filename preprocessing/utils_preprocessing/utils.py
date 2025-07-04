import os
import pandas as pd
import argparse
import unicodedata
import re
import string
import csv
import pickle
import numpy as np
from collections import Counter

french_stopwords = set([
    "alors", "au", "aucuns", "aussi", "autre", "avant", "avec", "avoir", "bon", "car", "ce", "cela", "ces", "ceux", "chaque", "ci", "comme", "comment", "dans", "des", "du", "dedans", "dehors", "depuis", "devrait", "doit", "donc", "dos", "droite", "début", "elle", "elles", "en", "encore", "essai", "est", "et", "eu", "fait", "faites", "fois", "font", "force", "haut", "hors", "ici", "il", "ils", "je", "juste", "la", "le", "les", "leur", "là", "ma", "maintenant", "mais", "mes", "mine", "moins", "mon", "mot", "même", "ni", "nommés", "notre", "nous", "nouveaux", "ou", "où", "par", "parce", "parole", "pas", "personnes", "peut", "peu", "pièce", "plupart", "pour", "pourquoi", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa", "sans", "ses", "seulement", "si", "sien", "son", "sont", "sous", "soyez", "sujet", "sur", "ta", "tandis", "tellement", "tels", "tes", "ton", "tous", "tout", "trop", "très", "tu", "voient", "vont", "votre", "vous", "vu", "ça", "étaient", "état", "étions", "été", "être", "mmol","mol","nmol"
])

#import spacy
#nlp = spacy.load("fr_core_news_sm")
#french_stopwords = set(nlp.Defaults.stop_words).union(french_stopwords)
#del nlp

french_stopwords = ['a','ai','aie','aient','aies','ait','alors','as','au','aucun','aura','aurai','auraient','aurais','aurait','auras','aurez','auriez','aurions','aurons','auront','aussi','autre','aux','avaient','avais','avait','avant','avec','avez','aviez','avions','avoir','avons','ayant','ayez','ayons','bon','car','ce','ceci','cela','ces','cet','cette','ceux','chaque','ci','comme','comment','d','dans','de','dedans','dehors','depuis','des','deux','devoir','devrait','devrez','devriez','devrions','devrons','devront','dois','doit','donc','dos','droite','du','dès','début','dù','elle','elles','en','encore','es','est','et','eu','eue','eues','eurent','eus','eusse','eussent','eusses','eussiez','eussions','eut','eux','eûmes','eût','eûtes','faire','fais','faisez','fait','faites','fois','font','force','furent','fus','fusse','fussent','fusses','fussiez','fussions','fut','fûmes','fût','fûtes','haut','hors','ici','il','ils','j','je','juste','l','la','le','les','leur','leurs','lui','là','m','ma','maintenant','mais','me','mes','moi','moins','mon','mot','même','n','ne','ni','nom','nommé','nommée','nommés','nos','notre','nous','nouveau','nouveaux','on','ont','ou','où','par','parce','parole','pas','personne','personnes','peu','peut','plupart','pour','pourquoi','qu','quand','que','quel','quelle','quelles','quels','qui','sa','sans','se','sera','serai','seraient','serais','serait','seras','serez','seriez','serions','serons','seront','ses','seulement','si','sien','soi','soient','sois','soit','sommes','son','sont','sous','soyez','soyons','suis','sujet','sur','t','ta','tandis','te','tellement','tels','tes','toi','ton','tous','tout','trop','très','tu','un','une','valeur','voient','vois','voit','vont','vos','votre','vous','vu','y','à','ça','étaient','étais','était','étant','état','étiez','étions','été','étés','êtes','être']

def remove_symbols(text):
    # Remove everything that's not a letter, digit, or whitespace
    return re.sub(r'[^\w\s]', '', text)

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

def get_notes(file_path,column='input', column_codes='labels', labels=False):
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
                true_codes = row[column_codes].strip().split()
                true_codes = [remove_symbols(code) for code in true_codes]
                codes.append(true_codes)
        print('Documents loaded')

    if labels:
        return texts, codes
    else:
        return texts

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
          
    #for word, count in word_counts.items():
    #    norm_word = normalize_text(word)
    #    if norm_word == 'mmol':
    #        print(f"Original: {word}, Normalized: {norm_word}, Count: {count}")
    #        print(f"In range: {lower_threshold <= count <= upper_threshold}")
    #        print(f"In stops: {norm_word in stops}")
    #        print(f"Length > 2: {len(norm_word) > 2}")
            
    print('Len vocab: ', len(percentile_vocab))
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

