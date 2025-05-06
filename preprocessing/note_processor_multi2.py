import csv
import pickle
from multiprocessing import Pool, Lock
from keywords_extraction2 import KeywordsExtractor
import spacy
from tqdm import tqdm
import time
import argparse

import json

# List of common French negation words
negations = {"ne", "pas", "jamais", "n'", "n’", "non", "rien", "personne", "aucun"}

def contains_negation(sentence):
    """Function to check if a sentence contains a negation."""
    return any(token.lower_ in negations for token in sentence)

def initialize_nlp(args):
    """Initialize the spaCy model and global matcher."""
    dictionary_path, max_window, threshold = args
    global nlp, extractor
    nlp = spacy.load("fr_core_news_sm")
    with open(dictionary_path, 'rb') as file:
        definitions = pickle.load(file)
    definitions = definitions.keys()
    extractor = KeywordsExtractor(text_path=None, list_definitions=definitions, max_window=max_window, threshold=threshold)

def process_line(row):
    """Remove negated sentences and extract keywords."""
    # Clean and parse text
    doc = nlp(row[-2].replace('\n', ' '))

    # Filter out sentences with negations in a generator
    filtered_text = ' '.join(sent.text for sent in doc.sents if not contains_negation(sent))

    # Extract keywords
    keywords = [match['match'] for match in extractor.extract(filtered_text)]

    # Construct new row
    return row[:-2] + [keywords, row[-1]]

def count_lines_in_csv(file_path):
    """Count the number of lines in a CSV file efficiently"""
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return sum(1 for row in reader)

def line_generator(filename):
    """Generator to yield lines from a file."""
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row

def line_generator2(filename):
    """Generator to yield lines from a file."""
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        yield from csv.reader(f)

def line_generator3(filename, chunksize=20):
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        for row in chunk.itertuples(index=False, name=None):
            yield list(row)

def write_results(results, output_file_path, lock):
    """Write results to the output file."""
    with lock:
        with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for result in results:
                writer.writerow(result)

def main(note_path, dictionary_path, output_file_path,max_window=5,threshold=0.8):
    t1 = time.time()
    NUM_WORKERS = 4  # Limit the number of parallel processes
    
    lock = Lock()
    number_lines = count_lines_in_csv(note_path)
    print(number_lines)
    
    args = (dictionary_path,max_window,threshold)
    # Initialize the pool with the global matcher
    with Pool(processes=NUM_WORKERS, initializer=initialize_nlp, initargs=(args,)) as pool:
        # Process lines in parallel
        results = list(tqdm(pool.imap_unordered(process_line, (row for row in line_generator2(note_path) if row[2].isdigit() and int(row[2])> 5 )), total=number_lines, desc="Processing lines"))
        
        #results = list(tqdm(pool.imap_unordered(process_line, (row for row in line_generator3(note_path) if row[2] and row[2] > 5 )), total=number_lines, desc="Processing lines"))

    t2 = time.time()
    print('Total time: ', (t2-t1)/60)
    # Write results to the output file
    write_results(results, output_file_path, lock)
    return t2-t1
    
if __name__ =='__main__':
    
    #parser = argparse.ArgumentParser(descrption= 'Setting note dictionary and output path')
    #parser.add_argument(note_path, type=str, help='Path to the notes')
    #parser.add_argument(dictionary_path, type=str, help='Path to dictionary for keywords extraction')
    #parser.add_argument(output_file_path, type=str, help='Output file path')
    
    #args = parser.parse_args()
    #main(note_path=args.note_path, dictionary_path=args.dictionary_path, output_file_path=args.output_file_path)
    
    #note_path = args.note_path
    #dict_path = args.dictionary_path
    #output_path = args.output_file_path
    
    thresholds = [0.7]
    windows = [5]
    results = {}
    
    for th in thresholds:
        for w in windows:
            print(f'Running: th={th} window={w}')
            output_path = '../results_sweep/results_' + str(th) + '_' + str(w)+'.csv'
            t = main(note_path='../../data/crh_omop_2024/test_1000/train.csv', dictionary_path='../aphp_final_no_single_letter.pkl', output_file_path=output_path, max_window=w, threshold=th)
            results[(th,w)] = t
    
    with open('../results_sweep/time.json', 'w') as f:
        json.dump(results, f)
    
            
    #main(note_path='../../data/crh_omop_2024/all/test.csv', dictionary_path='../aphp_final.pkl', output_file_path='../prova.csv')
