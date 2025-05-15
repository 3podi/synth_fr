import os
import spacy
from pathlib import Path
from typing import List, Union
import shutil
import tempfile

from utils_preprocessing.utils import normalize_text
import pysimstring.simstring as simstring


# Load spaCy model
nlp = spacy.load("fr_core_news_sm")

# Constants
VALID_POS_START_END = {"CONJ", "ADP", "DET"}
MAX_WINDOW = 5


class SimstringWriter:
    def __init__(self, db_path):
        self.db_path = db_path
        self.writer = None

    def __enter__(self):
        self.writer = simstring.writer(self.db_path, 3, False, True)
        return self.writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

class KeywordsExtractor:
    def __init__(
        self,
        database_dir: str = None,
        text_path: str = None,
        list_definitions: List[str] = None,
        max_window: int = MAX_WINDOW,
        threshold: float = 0.8,
        similarity_measure: str = "jaccard"
    ):
        """
        Extracts keyword-like spans from input text using approximate matching via SimString.

        Args:
            database_dir (str): Directory where the SimString database is stored.
            text_path (str): Text file containing keywords (one per line).
            list_definitions (List[str]): Additional keywords.
            max_window (int): Max span length.
            threshold (float): Similarity threshold.
            similarity_measure (str): One of "cosine", "jaccard", "dice", etc.
        """
        assert database_dir or text_path or list_definitions, \
            "You must provide either a database_dir or some keyword sources"

        self.max_window = max_window
        self.threshold = threshold
        self.similarity_measure = similarity_measure

        db_file = "terms.simstring"

        # If no database_dir provided, create default folder
        if database_dir is None:
            database_dir = os.path.join(os.getcwd(), "simstring_db")
            
            # Remove the folder if it already exists
            if os.path.exists(database_dir):
                shutil.rmtree(database_dir)
            
            os.makedirs(database_dir, exist_ok=True)
            self.build_database(os.path.join(database_dir, db_file), text_path, list_definitions)
        else:
            if not os.path.exists(os.path.join(database_dir, db_file)):
                self.build_database(os.path.join(database_dir, db_file), text_path, list_definitions)


        self.reader = simstring.reader(os.path.join(database_dir, db_file))
        self.reader.measure = getattr(simstring, similarity_measure)
        self.reader.threshold = threshold

    def build_database(self, db_path: str, text_path: str = None, list_definitions: List[str] = None):
        """
        Builds the SimString database from text or list of terms.
        """
        with SimstringWriter(db_path) as db:
            if text_path:
                with open(text_path, "r", encoding="utf-8") as f:
                    print('Building from .txt')
                    for line in f:
                        term = line.strip()
                        if term:
                            db.insert(normalize_text(term))
            if list_definitions:
                print('Building from definitions')
                for term in list_definitions:
                    if term:
                        #doc = nlp(term)
                        #lemmas = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
                        #term = ' '.join(lemmas).strip()
                        db.insert(normalize_text(term))

    def extract(self, text: str):
        doc = nlp(text)
        spans = self.generate_valid_spans(doc)
        raw_matches = self.match_spans(spans)
        return self.resolve_overlaps(raw_matches)

    def generate_valid_spans(self, doc, window_size=None):
        window_size = window_size or self.max_window
        valid_spans = []
        for sent in doc.sents:
            for i in range(len(sent)):
                for j in range(i, min(i + window_size, len(sent))):
                    span = sent[i:j + 1]
                    if self.is_valid_sequence(span):
                        valid_spans.append(span)
        return valid_spans

    def is_valid_sequence(self, span):
        tokens = list(span)
        if not tokens or len(tokens) > self.max_window:
            return False
        if len(tokens) == 1:
            tok = tokens[0]
            if tok.is_stop or tok.like_num:
                return False
        if tokens[0].pos_ in VALID_POS_START_END or tokens[-1].pos_ in VALID_POS_START_END:
            return False
        if tokens[0].is_punct:
            return False
        return True

    def match_spans(self, spans: List[spacy.tokens.Span]):
        matches = []
        for span in spans:
            text = normalize_text(span.text)
            if not text:
                continue
            candidates = self.reader.retrieve(text)
            for c in candidates:
                matches.append({
                    "span": span,
                    "match": c,
                    "similarity": 1.0  # SimString reader doesn't provide score
                })
        return matches

    def resolve_overlaps(self, matches: List[dict]):
        final_matches = []
        used_tokens = set()

        matches = sorted(matches, key=lambda m: -(m["span"].end - m["span"].start))

        for match in matches:
            idxs = set(range(match["span"].start, match["span"].end))
            if used_tokens.intersection(idxs):
                continue
            final_matches.append(match)
            used_tokens.update(idxs)

        return final_matches

def main():
    input_text = 'The patient suffer of Vertigo and nausea and Migraine and Lantus (Insulin Glargine).'

    extractor = KeywordsExtractor(
        text_path='preprocessing/strings.txt'
    )
    results = extractor.extract(input_text)

    print('Final matches:')
    for r in results:
        print(f"- Span: '{r['span']}' | Match: '{r['match']}'")

    print('Number of matches:', len(results))


if __name__ == '__main__':
    main()
