import pandas as pd
import argparse
import os
import ast
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

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

def main(results_folder, dictionary_path):
    extractions = [f for f in os.listdir(results_folder) if f.endswith('.csv')]

    with open(dictionary_path, 'rb') as f:
        corpus = pickle.load(f)
    
    corpus = {k.lower(): v for k, v in corpus.items()}

    for extr in extractions:
        path = os.path.join(results_folder, extr)
        df = pd.read_csv(path)

        tp_list = []
        fp_list = []
        fn_list = []

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
                true_codes = [code[0] for code in true_codes]
            else:
                true_codes = []

            # Handle case with no predictions
            predicted_codes = [corpus[match] for match in predicted_expressions if match in corpus]
            predicted_codes = [
                corpus[match][0]
                for match in predicted_expressions
                if match in corpus and corpus[match] and corpus[match][0] is not None
            ]
            
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
        records.append((param1, param2, precision, recall, f1))

    # Convert to DataFrame for plotting
    results_df = pd.DataFrame(records, columns=['param1', 'param2', 'precision', 'recall', 'f1'])

    # Pivot for heatmap
    heatmap_data = results_df.pivot(index='param2', columns='param1', values='f1')

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='viridis')
    plt.title("F1 Score Heatmap")
    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")
    plt.tight_layout()
    # Save the plot
    save_path = os.path.join(results_folder, "f1_heatmap.png")
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument('results_folder', type=str, help='Path to the result folder')
    parser.add_argument('dictionary_path', type=str, help='Path to the dictionary (pickle format)')
    parser.add_argument('--total_metrics_only', action='store_true', help='Optionally compute final metrics only')
    args = parser.parse_args()

    if not args.total_metrics_only:
        main(results_folder=args.results_folder, dictionary_path=args.dictionary_path)
    else:
        total_metrics(results_folder=args.results_folder)