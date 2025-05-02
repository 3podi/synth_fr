import csv
import pickle
from multiprocessing import Pool, Manager
from keywords_extraction import KeywordsExtractor
import spacy
from tqdm import tqdm

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

def process_line(row, extractor):
    """Process the line to remove sentences with negations and extract keywords."""
    text = row[-2].replace('\n', ' ')
    doc = nlp(text)

    # Extract sentences and check for negations
    sentences_with_no_negations = [sent.text for sent in doc.sents if not contains_negation(sent)]
    text = " ".join(sentences_with_no_negations)

    # Extract keywords
    results = process_note(nlp(text), extractor)
    results = [match['match'] for match in results]

    # Prepare the new row
    new_row = row[:-2]
    new_row += [results]
    new_row += [row[-1]]

    return new_row

def line_generator(filename):
    """Generator to yield lines from a file."""
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row

def write_results(results, output_file_path):
    """Write results to the output file."""
    with open(output_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for result in results:
            writer.writerow(result)

def main(note_path, dictionary_path, output_file_path):
    NUM_WORKERS = 4  # Limit the number of parallel processes

    # Create a manager to share the extractor across processes
    manager = Manager()
    shared_extractor = manager.Namespace()
    shared_extractor.extractor = None

    # Initialize the pool with the global matcher
    with Pool(processes=NUM_WORKERS, initializer=initialize_nlp, initargs=(dictionary_path,)) as pool:
        # Process lines in parallel
        results = pool.starmap(process_line, [(row, shared_extractor.extractor) for row in tqdm(line_generator(note_path), desc="Processing lines") if row[2].isdigit() and int(row[2]) > 5])

    # Write results to the output file
    write_results(results, output_file_path)

if __name__ =='__main__':
    main(note_path='../../data/crh_omop_2024/test_1000/test.csv', dictionary_path='../aphp_final.pkl', output_file_path='../prova.csv')