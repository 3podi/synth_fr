from keywords_extraction import KeywordsExtractor
import spacy
import pickle
from tqdm import tqdm
import csv

# Load the French spaCy model
nlp = spacy.load("fr_core_news_sm")

# List of common French negation words
negations = {"ne", "pas", "jamais", "n'", "n’", "non", "rien", "personne", "aucun"}
def contains_negation(sentence):
    # Function to check if a sentence contains a negation
    return any(token.lower_ in negations for token in sentence)

def line_generator(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line  # yields one line at a time

def process_note(note, matcher):
    """note: must be a spacy doc"""

    # Process the text with spaCy
    #doc = nlp(text)

    # Extract sentences and check for negations
    sentences_with_no_negations = [sent.text for sent in note.sents if not contains_negation(sent)]

    note = " ".join(sentences_with_no_negations)

    matches = matcher.extract(note)

    return matches


def main(note_path, dictionary_path, output_file_path):

    with open(dictionary_path, 'rb') as file:
        definitions = pickle.load(file)
    definitions = definitions.keys()

    extractor = KeywordsExtractor(text_path=None, list_definitions=definitions, max_window=5)   

    with open(note_path, 'r', newline='', encoding='utf-8') as f,  open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        reader = csv.reader(f)
        for i, row in tqdm(enumerate(reader)):
                        
            if i != 0 and int(row[2])>5:
                text = row[-2]
                text = text.replace('\n', ' ')
                results = process_note(nlp(text), extractor)

                results = [match['match'] for match in results]
                
                new_row = row[:-2]
                new_row += [results]
                new_row += [row[-1]]

                writer.writerow(new_row)


if __name__ =='__main__':
    main(note_path='../../data/crh_omop_2024/test_1000/test.csv', dictionary_path='../aphp_final.pkl', output_file_path='../prova.csv')