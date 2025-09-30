import os
import random
from typing import List

import pandas as pd
import ast
import unicodedata
import re
import argparse
import pickle
from collections import defaultdict
import string
from tqdm import tqdm

NUM_SAMPLES = 1000

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

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extracted_keywords_path", type=str, required=True, help="Path to a .csv file that has extracted keywords and ground truth icd-10 codes"
    )
    
    parser.add_argument(
        "--list_definitions_path", type=str, required=True,
        help="Path to extract list of codes to use and to generate code2int mapping"
    )

    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path of folder where to save the final parquet"
    )

    return parser.parse_args()

def remove_symbols(text):
    # Remove everything that's not a letter, digit, or whitespace
    return re.sub(r'[^\w\s]', '', text)

def get_reverse_dict(dictionary_path):
    
    with open(dictionary_path, 'rb') as f:
        corpus_no_norm = pickle.load(f)
    
    corpus = defaultdict(list)
    for k, v in corpus_no_norm.items():
        norm_k = normalize_text(k)    
        corpus[norm_k].append(k)
    return corpus

def get_code2int_dict(codes_samples):
    
    all_codes = set()
    
    for code_list in codes_samples:
        code_list = [remove_symbols(c) for c in code_list.strip().split()] if isinstance(code_list, str) else []
        all_codes.update(code_list)
    
    code2int = {c: i for i, c in enumerate(all_codes)}
    return code2int

def get_code2keys_dict(dictionary_path):
    with open(dictionary_path, 'rb') as f:
        corpus_no_norm = pickle.load(f)
        
    code_to_keys_inverse = defaultdict(list)
    for key, val in corpus_no_norm.items():
        if isinstance(val, list):
            for code in val:
                code_to_keys_inverse[remove_symbols(code)].append(key)
        else:
            code_to_keys_inverse[remove_symbols(val)].append(key)
    
    return code_to_keys_inverse

def map_codes2int(codes, code2int):
    
    integers = [code2int[remove_symbols(c)] for c in codes.strip().split()] if isinstance(codes,str) else []
    return " ".join(str(i) for i in integers)
        
def invert_norm_keywords(keywords, reverse_norm_dict):
    try:
        keywords = ast.literal_eval(keywords)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(keywords, list):
        return []

    return ", ".join([reverse_norm_dict.get(key, "")[0] for key in keywords]) #If more keys are mapped to the same normalization, take the 1st

def generate_grpo_dataset(input_path: str, output_dir: str, dictionary_path: str, seed: int = 42):
    """
    Generate grpo dataset with at least 2 coloumns 'keywords' and 'solution'.
    Keywors column is a list of keywords to use in the prompt.
    Solution coloumn is a sequence of icd-10 codes mapped to integer.

    Parameters:
        input_path (str): Path to the input .csv file.
        output_dir (str): Directory to save the dataset.
        dictionary_path (str): Path to the pkl with the vocabulary.
        seed (int): Random seed for reproducibility.
    """
    df = pd.read_csv(input_path)
    # Shuffle the full dataset reproducibly
    #shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    reverse_dict = get_reverse_dict(dictionary_path)
    code2int = get_code2int_dict(df.iloc[:,-1])
    df.iloc[:,-2] = df.iloc[:,-2].apply(lambda x: invert_norm_keywords(x, reverse_norm_dict=reverse_dict))
    df.iloc[:,-1] = df.iloc[:,-1].apply(lambda x: map_codes2int(x, code2int=code2int))
    
    # Rename using .columns
    df.columns.values[-2] = 'keywords'
    df.columns.values[-1] = 'solution'

 
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "grpo_dataset.parquet")
    df.to_parquet(output_path)
    
    output_path = os.path.join(output_dir, "code2int.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(my_dict, f)
    
def generate_grpo_dataset_random(input_path: str, output_dir: str, dictionary_path: str, seed: int = 42):
    """
    Generate RANDOM grpo dataset with at least 2 coloumns 'keywords' and 'solution'.
    Keywors column is a list of keywords to use in the prompt.
    Solution coloumn is a sequence of icd-10 codes mapped to integer.

    Parameters:
        input_path (str): Path to the input .csv file.
        output_dir (str): Directory to save the dataset.
        dictionary_path (str): Path to the pkl with the vocabulary.
        seed (int): Random seed for reproducibility.
    """
    df = pd.read_csv(input_path)
    # Shuffle the full dataset reproducibly
    #shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    code2key = get_code2keys_dict(dictionary_path)
    code2int = get_code2int_dict(df.iloc[:,-1])
    del df
    all_codes = list(code2int.keys())
    
    keywords = []
    solution = []
    real_codes = []
    number_codes = []
    number_keywords = []
    
    for i in tqdm(range(NUM_SAMPLES)):
        # Randomly sample a number of codes
        n_codes = random.randint(1, 10)
        # Randomly sample a number of keywords
        n_kws = random.randint(1, 3)
    
        sampled_codes = random.sample(all_codes, n_codes)
        
        sampled_kws = []
        for c in sampled_codes:
            mapped_kws = code2key[c]
            n_kws = min(n_kws,len(mapped_kws))
            sampled_kws.extend(random.sample(mapped_kws, n_kws))
            
        if len(sampled_kws) == 0:
            continue
        
        real_codes.append(" ".join(sampled_codes))
        integers = [code2int[c] for c in sampled_codes]

        keywords.append(", ".join(sampled_kws))
        solution.append(" ".join(str(c) for c in integers))
        assert n_codes == len(integers), 'Cannot find the integer number associated to some code'
        number_codes.append(n_codes)
        number_keywords.append(len(sampled_kws))

    data = {
        'keywords': keywords,
        'solution': solution,
        'codes': real_codes,
        'n_codes': number_codes,
        'n_kws': number_keywords
    }
    df = pd.DataFrame(data)
 
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "grpo_random_dataset.parquet")
    df.to_parquet(output_path)
    
    output_path = os.path.join(output_dir, "code2int.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(my_dict, f)
    
if __name__ == "__main__":
    
    args = parse_arguments()
    
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    generate_grpo_dataset_random(
        input_path=args.extracted_keywords_path,
        output_dir=output_dir,
        dictionary_path=args.list_definitions_path
    )
    

    
    
