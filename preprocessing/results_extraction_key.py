import pandas as pd
import argparse
import os
import ast
import pickle
import re
from collections import Counter
from utils_preprocessing.utils import normalize_text

import matplotlib.pyplot as plt
from collections import defaultdict

import spacy

import time

nlp = spacy.load("fr_core_news_sm")

def compute_confusion_matrix(y_pred, y_true):
    pred_set = set(y_pred)
    true_set = set(y_true)
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    return tp, fp, fn

def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

def remove_symbols(text):
    # Remove everything that's not a letter, digit, or whitespace
    return re.sub(r'[^\w\s]', '', text)

def main(results_folder, dictionary_path):
    extractions = [f for f in os.listdir(results_folder) if f.endswith('.csv')]

    with open(dictionary_path, 'rb') as f:
        corpus_no_norm = pickle.load(f)
    
    corpus = defaultdict(list)
    for k, v in corpus_no_norm.items():
        norm_k = normalize_text(k)
        #print(v[0])
        if norm_k not in corpus or v is not None:
            codes = list(set(remove_symbols(str(c)) for c in v))            
            corpus[norm_k]=codes  
    
    code_to_keys_inverse = defaultdict(list)

    for key, val in corpus.items():
        if isinstance(val, list):
            for code in val:
                code_to_keys_inverse[remove_symbols(code)].append(key)
        else:
            code_to_keys_inverse[remove_symbols(val)].append(key)

    key_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    #t1 = time.time()
    
    for extr in extractions:
        path = os.path.join(results_folder, extr)
        df = pd.read_csv(path)

        # Per-key TP, FP, FN counters
        
        i=0

        for idx, row in df.iterrows():
            if idx%1000 == 0:
                print(idx)
                #t2 = time.time()
                #print('Time x row: ', (t2-t1)/1000)
                #print(ciao)
            try:
                predicted_expressions = ast.literal_eval(row.iloc[-2])
                if not isinstance(predicted_expressions, list):
                    predicted_expressions = []
            except (ValueError, SyntaxError):
                predicted_expressions = []

            true_codes_raw = row.iloc[-1]
            true_codes = [remove_symbols(c) for c in true_codes_raw.strip().split()] if isinstance(true_codes_raw, str) else []
            

            # Build code -> keys mapping (reverse lookup)
            code_to_keys = defaultdict(list)
            pred_codes = set()
            
            norm_predicted_expressions = list(set(normalize_text(expr) for expr in predicted_expressions))
            for expr in norm_predicted_expressions:
                mapped = corpus.get(expr)
                if mapped:
                    if isinstance(mapped, list):
                        for code in mapped:
                            clean_code = remove_symbols(code)
                            code_to_keys[clean_code].append(expr)
                            pred_codes.add(clean_code)
                    else:
                        clean_code = remove_symbols(mapped)
                        code_to_keys[clean_code].append(expr)
                        pred_codes.add(clean_code)

            true_set = set(true_codes)

            # TP: predicted and correct
            for code in pred_codes & true_set:
                for key in code_to_keys[code]:
                    key_stats[key]['tp'] += 1

            # FP: predicted but not correct
            for code in pred_codes - true_set:
                for key in code_to_keys[code]:
                    key_stats[key]['fp'] += 1

            # FN: correct but not predicted
            #for code in true_set - pred_codes:
            #    keys_that_map = code_to_keys_inverse.get(code, [])
            #    for key in keys_that_map:
            #        key_stats[key]['fn'] += 1
            for code in true_set:
                keys_that_map = code_to_keys_inverse.get(code, [])
                for key in keys_that_map:
                    if key not in norm_predicted_expressions:
                        key_stats[key]['fn'] += 1

    # Save per-key metrics

    per_key_metrics = []
    for key, stats in key_stats.items():
        precision, recall, f1 = compute_metrics(stats['tp'], stats['fp'], stats['fn'])

        per_key_metrics.append({
            'dictionary_keys': key,
            'code': corpus[normalize_text(key)],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': stats['tp'],
            'fp': stats['fp'],
            'fn': stats['fn'],
        })
        
    metrics_folder = os.path.join(results_folder, "with_metrics")
    os.makedirs(metrics_folder, exist_ok=True)

    metrics_df = pd.DataFrame(per_key_metrics)
    metrics_df = metrics_df.sort_values(by='f1', ascending=False)
    metrics_df.to_csv(os.path.join(metrics_folder, extr.replace('.csv', '_metrics_per_key.csv')), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument('results_folder', type=str, help='Path to the result folder')
    parser.add_argument('dictionary_path', type=str, help='Path to the dictionary (pickle format)')
    args = parser.parse_args()

    main(results_folder=args.results_folder, dictionary_path=args.dictionary_path)

