import pandas as pd
import csv
from collections import Counter
import argparse
from tqdm import tqdm
import pickle
from pathlib import Path
import re

def process_large_csv(file_path, column):
    for chunk in pd.read_csv(file_path, chunksize=1, usecols=[column]):
        for value in chunk[column]:
            if pd.notnull(value):
                yield value


def count_lines_in_csv(file_path):
    """Count the number of lines in a CSV file efficiently"""
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return sum(1 for row in reader) - 1


def capital_ratio(word, threshold=0.7):
    """Return True if the ratio of capital letters in word exceeds threshold."""
    if len(word) == 0:
        return False
    upper_count = sum(1 for c in word if c.isupper())
    return (upper_count / len(word)) >= threshold

# Main function to find candidate acronyms
def find_acronyms(text, cap_ratio_threshold=0.7):
    patterns = [
        r'\b[A-Z]{2,}\b',                        # All-caps
        r'\b(?:[A-Z]\.){2,}',                    # Dotted
        r'\((?P<acronym>[A-Z]{2,})\)',           # Parentheses
        r'\b(?:[A-Z]{2,}-)+[A-Z]{2,}\b',         # Hyphenated
        r'\b(?:[A-Z][a-z]+\s)+\([A-Z]{2,}\)',    # Long-form followed by acronym
    ]

    matches = []

    # Apply regex rules
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                match = match[0]
            matches.append(match)

    # Tokenize and apply capital ratio rule
    #words = text.split()
    words = re.findall(r"[A-Za-z\-\.]+", text)
    for word in words:
        if capital_ratio(word, threshold=cap_ratio_threshold):
            matches.append(word)

    return matches


def abbreviations_searcher(file, column, cap_ratio=0.5):
    """
    Count abbreviations in a CSV file

    Args:
        file (str): Path to the CSV file.
        column (str): Name of the column containing text.
        cap_ratio (float): capital letters percentage
    """

    # Count total lines for a nice progress bar (can slow things down)
    total_lines = count_lines_in_csv(file)
    chunksize = 1000

    # Second pass: Compute n-grams with limited vocabulary
    counter = Counter()
    with tqdm(total=total_lines, desc="Processing rows") as pbar:
        for chunk in pd.read_csv(file, chunksize=chunksize, usecols=[column]):
            for text in chunk[column].dropna():
                matches = find_acronyms(text,cap_ratio_threshold=cap_ratio)
                counter.update(matches)
                pbar.update(1)

    return counter


def main(file_path,column,cap_ratio,save_flag,top_k):

    counter = abbreviations_searcher(file_path,column,cap_ratio)

    print("Top 10 abbreviations:")
    for k, v in counter.most_common(top_k):
        print(" ".join(k), ":", v)
        
    # Save
    if save_flag:
        output_dir = Path("preprocessing/frequency_analysis/results_frequency")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"abbreviations_counter.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(counter, f)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description= 'Setting paths and params')
    parser.add_argument('file_path', type=str, help='Path to the text file')
    parser.add_argument('column', type=str, help='Name of the text column')
    parser.add_argument('--cap_ratio', type=float, default=0.5, help='Name of the text column')
    parser.add_argument('--save', action='store_true', help='Optionally save n-gram count')
    parser.add_argument('--top_k', type=int, default=10, help='Number of results to print')

    args=parser.parse_args()

    main(args.file_path, args.column,args.cap_ratio,args.save,args.top_k)