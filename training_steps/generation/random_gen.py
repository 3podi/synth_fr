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

from vllm import LLM, SamplingParams
import csv

PROMPT = """### Instruction
Vous agissez en tant qu'IA médicale spécialisée. Votre tâche consiste à rédiger un rapport d'hospitalisation fictif complet et réaliste. Le document doit présenter une évolution clinique cohérente avec une terminologie médicale précise, tout en respectant scrupuleusement la séquence imposée des mots-clés. Retournez uniquement le rapport.
### Keywords
{keywords}
### Output
"""

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
        "--csv_path", type=str, required=True,
        help="Path to extract list of codes to use and to generate code2int mapping"
    )

    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path of folder where to save the final parquet"
    )
    
    parser.add_argument(
        "--max_codes", type=int, default=1,
        help="Max number of codes to randomly sample for each generation"
    )
    
        parser.add_argument(
        "--max_kws", type=int, default=1,
        help="Max number of code definition to keep for each code"
    )
        
        parser.add_argument(
        "--num_samples", type=int, default=10,
        help="Number of samples to generate"
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

def get_code2keys_dict(csv_path):
    """
    Reads a CSV file and returns a dictionary mapping codes to keys.
    
    Args:
        csv_path (str): Path to the CSV file with 'dictionary_keys' and 'code' columns
        
    Returns:
        dict: Dictionary mapping codes to lists of keys
    """
    code_to_keys_inverse = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row['dictionary_keys']
            code = row['code']
            code_to_keys_inverse[remove_symbols(code)].append(key)
    
    # Convert defaultdict to regular dict for cleaner output
    return dict(code_to_keys_inverse)

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


def sample_keywords(csv_path: str, num_samples: int = 10, max_codes: int = 1, max_kws: int = 1, seed: int = 42):
    """
    Generate RANDOM grpo dataset with at least 2 coloumns 'keywords' and 'codes'.
    Keywords column is a list of keywords to use in the prompt.
    Codes coloumn is a sequence of icd-10 codes.

    Parameters:
        csv_path (str): Path to the pkl with the vocabulary.
        num_samples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
    """
    
    code2key = get_code2keys_dict(csv_path)
    all_codes = list(code2key.keys())
    
    keywords = []
    codes = []
    number_codes = []
    number_keywords = []
    
    for i in tqdm(range(num_samples)):
        # Randomly sample a number of codes
        n_codes = random.randint(1, max_codes)
        # Randomly sample a number of keywords
        N_kws = random.randint(1, max_kws)
    
        sampled_codes = random.sample(all_codes, n_codes)
        
        sampled_kws = []
        for c in sampled_codes:
            mapped_kws = code2key[c]
            n_kws = min(N_kws,len(mapped_kws))
            sampled_kws.extend(random.sample(mapped_kws, n_kws))
            
        if len(sampled_kws) == 0:
            continue
        
        codes.append(" ".join(sampled_codes))
        keywords.append(", ".join(sampled_kws))
        number_codes.append(n_codes)
        number_keywords.append(len(sampled_kws))

    data = {
        'keywords': keywords,
        'codes': codes,
        'n_codes': number_codes,
        'n_kws': number_keywords
    }
    df = pd.DataFrame(data)    
    return df

def make_regex():
    return (r".*(Motif d'hospitalisation).*?(Condition principal d'admission).*?(Antécédent).*?(Mode de vie).*?(Histoire de la maladie).*?(Examen clinique).*?(Examens complémentaires).*?(Évolution pendant l'hospitalisation).*?(Conclusion).*?")

def generate_responses(keywords, model_name: str, tp: int = 1, pp: int = 1):
    
    # Initialize the LLM with your chosen model
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        #enable_chunked_prefill=True,
        gpu_memory_utilization=0.95,
        max_model_len=16384
    )
    print("Generating responses")
    all_responses = []
    
    
    # Default gen configs of the model 
    sampling_params = SamplingParams(
        temperature=0.2,
        #top_k = 64,
        #top_p = 0.95,
        max_tokens=3000,
        #seed=random.randint(0, 2**32 - 1),
        #stop=["</s>"],
        #stop_token_ids = stop_token_ids,
        #presence_penalty=1.0,
        #frequency_penalty=1.2,
        n=1,
    )
    
    prompts = [ PROMPT.replace("{keywords}", kws) for kws in keywords]

    response = llm.generate(
        prompts,
        sampling_params=sampling_params,
        #guided_options_request=dict(guided_regex=make_regex())
    )
    all_responses.extend([output.outputs[0].text for output in response])


    return all_responses


if __name__ == "__main__":
    
    args = parse_arguments()
    
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    df = sample_keywords(
        csv_path=args.csv_path,
        num_samples=args.num_samples,
        max_codes=args.max_codes,
        max_kws=args.max_kws
    )
    
    responses = generate_responses(
        df['keywords'],
        model_name = "google/medgemma-27b-text-it" 
    )
    
    df['response'] = responses
    
    output_path = os.path.join(output_dir, "random_dataset.parquet")
    df.to_parquet(output_path)

    