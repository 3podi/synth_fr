import csv
import pickle
from multiprocessing import Pool, Lock
from keywords_extraction import KeywordsExtractor
import spacy
from tqdm import tqdm
import time
from itertools import islice

# List of common French negation words
negations = {"ne", "pas", "jamais", "n'", "n’", "non", "rien", "personne", "aucun"}

def contains_negation(sentence):
    """Function to check if a sentence contains a negation."""
    return any(token.lower_ in negations for token in sentence)

def initialize_nlp(dictionary_path):
    """Initialize the spaCy model and global matcher."""
    global nlp, extractor
    nlp = spacy.load("fr_core_news_sm")
    with open(dictionary_path, 'rb') as file:
        definitions = pickle.load(file)
    definitions = definitions.keys()
    extractor = KeywordsExtractor(text_path=None, list_definitions=definitions)

def process_batch(batch):
    """Process a batch of lines to remove sentences with negations and extract keywords."""

    texts = (row[-2].replace('\n', ' ') for row in batch)
    docs = nlp.pipe(texts)

    processed_rows = []
    for row, doc in zip(batch, docs):
        filtered_text = ' '.join(sent.text for sent in doc.sents if not contains_negation(sent))
        keywords = [match['match'] for match in extractor.extract(filtered_text)]
        new_row = row[:-2] + [keywords, row[-1]]
        processed_rows.append(new_row)

    return processed_rows

def batch_generator(filename, batch_size):
    """CSV reader with batching via islice."""
    with open(filename, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        while True:
            batch = list(islice(reader, batch_size))
            if not batch:
                break
            yield batch

def write_results(results, output_file_path, lock):
    """Write results to the output file with a lock for thread safety."""
    with lock:
        with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for result in results:
                writer.writerow(result)

def count_lines_in_csv(file_path):
    """Count the number of lines in a CSV file efficiently"""
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        return sum(1 for row in reader)

#def get_optimal_workers():
#    """Determine the optimal number of worker processes."""
#    cpu_count = multiprocessing.cpu_count()
#    # You can adjust this based on your specific needs
#    optimal_workers = max(1, cpu_count - 1)  # Leave one core free for other processes
#    return optimal_workers

def main(note_path, dictionary_path, output_file_path, batch_size=100):
    # Determine the optimal number of workers
    t1 = time.time()
    NUM_WORKERS = 4

    # Count the total number of lines to process
    total_lines = count_lines_in_csv(note_path)

    # Create a lock for thread-safe writing to the output file
    lock = Lock()

    # Initialize the pool with the global matcher
    with Pool(processes=NUM_WORKERS, initializer=initialize_nlp, initargs=(dictionary_path,)) as pool:
        # Process batches of lines in parallel with tqdm progress bar
        results = []
        for batch_results in tqdm(pool.imap_unordered(process_batch, batch_generator(note_path, batch_size)), total=total_lines // batch_size, desc="Processing lines"):
            results.extend(batch_results)
    
    t2 = time.time()
    print('Execution time: ', (t2-t1)/60)
    # Write results to the output file
    write_results(results, output_file_path, lock)

if __name__ =='__main__':
    main(note_path='../../data/crh_omop_2024/test_1000/test.csv', dictionary_path='../aphp_final.pkl', output_file_path='../prova.csv')