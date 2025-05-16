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

def main(results_folder, dictionary_path, compute_per_code_metrics=False, digits=None):
    extractions = [f for f in os.listdir(results_folder) if f.endswith('.csv')]

    with open(dictionary_path, 'rb') as f:
        corpus = pickle.load(f)
    
    corpus = {normalize_text(k): v for k, v in corpus.items()}
    
    for extr in extractions:
        path = os.path.join(results_folder, extr)
        df = pd.read_csv(path)

        tp_list = []
        fp_list = []
        fn_list = []

        # Per-code TP, FP, FN counters
        if compute_per_code_metrics:
            code_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        i=0
        for idx, row in df.iterrows():
            predicted_expressions = row.iloc[-2]
            true_codes = row.iloc[-1]
            
            # Convert stringified list to actual list safely
            try:
                predicted_expressions = ast.literal_eval(predicted_expressions)
                if not isinstance(predicted_expressions, list):
                    predicted_expressions = []
            except (ValueError, SyntaxError):
                predicted_expressions = []
           
            # Process true codes
            if isinstance(true_codes, str):
                true_codes = true_codes.strip().split()
                if digits:
                    true_codes = [remove_symbols(code)[:1+digits] for code in true_codes]
                else:
                    true_codes = [remove_symbols(code) for code in true_codes]
            else:
                true_codes = []

            # Handle case with no predictions
            if digits:
                predicted_codes = [
                    remove_symbols(corpus[match])[:1+digits]
                    for match in predicted_expressions
                    if corpus[match]
                ]
            else:
                
                              
                predicted_codes = [
                    remove_symbols(corpus[match])
                    for match in predicted_expressions
                    if corpus[match]
                ]

            # Per-code metrics
            if compute_per_code_metrics:
                pred_set = set(predicted_codes)
                true_set = set(true_codes)

                for code in pred_set:
                    if code in true_set:
                        code_stats[code]['tp'] += 1
                    else:
                        code_stats[code]['fp'] += 1
                for code in true_set:
                    if code not in pred_set:
                        code_stats[code]['fn'] += 1
            
            tp, fp, fn = compute_confusion_matrix(predicted_codes, true_codes)
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)

        df['TP'] = tp_list
        df['FP'] = fp_list
        df['FN'] = fn_list

        metrics_folder = os.path.join(results_folder, "with_metrics")
        os.makedirs(metrics_folder, exist_ok=True)

        new_filename = extr.replace('.csv', '_metrics.csv')
        new_path = os.path.join(metrics_folder, new_filename)
        df.to_csv(new_path, index=False)

        # Save per-code metrics if requested
        if compute_per_code_metrics:

            inverted_corpus = defaultdict(list)
            for expr, code in corpus.items():
                clean_code = remove_symbols(str(code))
                inverted_corpus[clean_code].append(expr)

            per_code_metrics = []
            for code, stats in code_stats.items():
                precision, recall, f1 = compute_metrics(stats['tp'], stats['fp'], stats['fn'])
                original_keys = "; ".join(inverted_corpus.get(code, []))  # combine if multiple keys map to same code
                per_code_metrics.append({
                    'dictionary_keys': original_keys,
                    'code': code,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': stats['tp'],
                    'fp': stats['fp'],
                    'fn': stats['fn'],
                })

            metrics_df = pd.DataFrame(per_code_metrics)
            metrics_df = metrics_df.sort_values(by='f1', ascending=False)
            metrics_df.to_csv(os.path.join(metrics_folder, extr.replace('.csv', '_metrics_per_code.csv')), index=False)

def total_metrics(results_folder):

    files = os.listdir(os.path.join(results_folder,'with_metrics'))
    files = [file for file in files if file.endswith('_metrics.csv')]

    for file in files:
        print(file)
        df = pd.read_csv(os.path.join(results_folder,'with_metrics',file))

        tp = sum(df['TP'].tolist())
        fp = sum(df['FP'].tolist())
        fn = sum(df['FN'].tolist())

        precision, recall, f1 = compute_metrics(tp, fp, fn)
        print(f'Precision: {precision} - Recall: {recall} - f1: {f1}')

def occurance_analysis(results_folder=None, dictionary_path=None):

    extractions = [f for f in os.listdir(results_folder) if f.endswith('.csv')]

    with open(dictionary_path, 'rb') as f:
        corpus = pickle.load(f)
    
    corpus = {normalize_text(k): v for k, v in corpus.items()}
    counter = Counter({element:0 for element in corpus.keys()})
    dict_codes = set(corpus.values())
    pred_codes = set()
    codes = set()
    total_words = 0
    total_predicted_words = 0

    for extr in extractions:
        path = os.path.join(results_folder, extr)
        df = pd.read_csv(path)

        for idx, row in df.iterrows():
            predicted_expressions = row.iloc[-2]
            total_words += int(row.iloc[2])
        
            try:
                predicted_expressions = ast.literal_eval(predicted_expressions)
                if not isinstance(predicted_expressions, list):
                    predicted_expressions = []
            except (ValueError, SyntaxError):
                predicted_expressions = []

            true_codes = row.iloc[-1]
            if isinstance(true_codes, str):
                true_codes = true_codes.strip().split()
                true_codes = [remove_symbols(code) for code in true_codes]
            else:
                true_codes = []

            predicted_codes = [
                    remove_symbols(corpus[match])
                    for match in predicted_expressions
                    if match in corpus and corpus[match]
                ]
            
            pred_codes.update(predicted_codes)
            codes.update(true_codes)
            counter.update(predicted_expressions)

            if not predicted_expressions:
                words = 0
            else:
                total_predicted_words += sum([len(expression.split()) for expression in predicted_expressions])

    missing_codes = codes - dict_codes
    zero_codes = dict_codes - pred_codes    
    zero_elements = [element for element, count in counter.items() if count == 0]
    print(f'Number of expression in dict never matched over total expressions: {len(zero_elements)}/{len(set(corpus.keys()))}')
    print(f'Number of codes never matched (several unmatched expressions can have same code) over total codes in dict: {len(zero_codes)}/{len(dict_codes)}')
    print(f'Number of codes not available in my dict but present in the data over total codes in data {len(missing_codes)}/{len(codes)}')
    print(f'Number codes not present in the data: {len(dict_codes-codes)}')

    print('Percentage extacted words: ', total_predicted_words/total_words * 100)

    

def total_metrics_plot(results_folder):
    metrics_path = os.path.join(results_folder, 'with_metrics')
    files = [f for f in os.listdir(metrics_path) if f.endswith('_metrics.csv')]

    records = []

    for file in files:
        try:
            param1 = float(file.split('_')[1])  # e.g., 0.9
            param2 = int(file.split('_')[2])    # e.g., 5
        except (IndexError, ValueError):
            print(f"Skipping file {file} due to unexpected naming format.")
            continue

        df = pd.read_csv(os.path.join(metrics_path, file))
        tp = df['TP'].sum()
        fp = df['FP'].sum()
        fn = df['FN'].sum()

        precision, recall, f1 = compute_metrics(tp, fp, fn)
        records.append((param1, param2, f1))

    # Convert to DataFrame for plotting
    results_df = pd.DataFrame(records, columns=['param1', 'param2', 'f1'])

    # Create the line plot
    plt.figure(figsize=(8, 6))
    for param2_val in sorted(results_df['param2'].unique()):
        subset = results_df[results_df['param2'] == param2_val].sort_values('param1')
        plt.plot(subset['param1'], subset['f1'], marker='o', label=f'param2 = {param2_val}')

    plt.xlabel("Similarity threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Similarity threshold")
    plt.legend(title="Max window")
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join(results_folder, "f1_vs_param1_by_param2.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument('results_folder', type=str, help='Path to the result folder')
    parser.add_argument('dictionary_path', type=str, help='Path to the dictionary (pickle format)')
    parser.add_argument('--total_metrics_only', action='store_true', help='Optionally compute final metrics only')
    parser.add_argument('--per_code_metrics', action='store_true', help='Optionally compute per code metrics' )
    parser.add_argument('--digits', type=int, default=None, help='Num digits after the code letter to consider for metrics')
    args = parser.parse_args()

    if not args.total_metrics_only:
        main(results_folder=args.results_folder, dictionary_path=args.dictionary_path, compute_per_code_metrics=args.per_code_metrics, digits=args.digits)
    else:
        total_metrics(results_folder=args.results_folder)
        occurance_analysis(results_folder=args.results_folder, dictionary_path=args.dictionary_path)