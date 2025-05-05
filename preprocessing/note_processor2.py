import csv
import pickle
import pandas as pd
import spacy
from tqdm import tqdm
from keywords_extraction import KeywordsExtractor

# List of common French negation words
negations = {"ne", "pas", "jamais", "n'", "n’", "non", "rien", "personne", "aucun"}

def contains_negation(sentence):
    """Function to check if a sentence contains a negation."""
    return any(token.lower_ in negations for token in sentence)

def process_note(row, nlp=None, extractor=None):
    """Process a single note to remove sentences with negations and extract keywords."""
    doc = nlp(row[-2].replace('\n', ' '))

    # Extract sentences and check for negations
    filtered_text = ' '.join(sent.text for sent in doc.sents if not contains_negation(sent))

    # Extract keywords
    keywords = [match['match'] for match in extractor.extract(filtered_text)]

    return row[:-2] + [keywords, row[-1]]

def write_results(results, output_file_path):
    """Write results to the output file."""
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

def main(note_path, dictionary_path, output_file_path):

    # Initialize the spaCy model and extractor
    nlp = spacy.load("fr_core_news_sm")
    with open(dictionary_path, 'rb') as file:
        definitions = pickle.load(file)
    definitions = definitions.keys()
    extractor = KeywordsExtractor(text_path=None, list_definitions=definitions)

    # Read the entire file into a DataFrame
    df = pd.read_csv(note_path)

    # Process each note sequentially
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing notes"):
        if row[2] > 5: 
            results.append(process_note(row, nlp, extractor))

    # Write results to the output file
    write_results(results, output_file_path)

if __name__ == '__main__':
    main(note_path='../../data/crh_omop_2024/test_1000/test.csv', dictionary_path='../aphp_final.pkl', output_file_path='../prova.csv')
    #main(note_path='../../data/crh_omop_2024/all/test.csv', dictionary_path='../aphp_final.pkl', output_file_path='../prova.csv')