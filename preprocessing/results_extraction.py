import pandas as pd
import argparse
import os
import pickle

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

    for extr in extractions:
        path = os.path.join(results_folder, extr)
        df = pd.read_csv(path)

        tp_list = []
        fp_list = []
        fn_list = []

        for idx, row in df.iterrows():
            predicted_expressions = row.iloc[-2]
            true_codes = row.iloc[-1]

            # Process true codes
            if isinstance(true_codes, str):
                true_codes = true_codes.strip().split()
            else:
                true_codes = []

            # Handle case with no predictions
            predicted_codes = []
            if isinstance(predicted_expressions, str) and predicted_expressions.strip():
                matches = predicted_expressions.strip().split()
                predicted_codes = [corpus[match] for match in matches if match in corpus]

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument('results_folder', type=str, help='Path to the result folder')
    parser.add_argument('dictionary_path', type=str, help='Path to the dictionary (pickle format)')
    args = parser.parse_args()

    main(results_folder=args.results_folder, dictionary_path=args.dictionary_path)