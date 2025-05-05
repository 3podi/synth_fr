import os
import spacy
from pathlib import Path
from typing import List, Union

import pysimstring.simstring as simstring


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Constants
VALID_POS_START_END = {"CONJ", "ADP", "DET"}
MAX_WINDOW = 5


class SimstringWriter:
    def __init__(self, path: Union[str, Path]):
        os.makedirs(path, exist_ok=True)
        self.path = path

    def __enter__(self):
        self.db_path = os.path.join(self.path, "terms.simstring")
        self.db = simstring.writer(self.db_path, 3, False, True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    def insert(self, term: str):
        self.db.insert(term)


class KeywordsExtractor:
    def __init__(
        self,
        database_dir: Union[str, Path] = None,
        text_path: str = None,
        list_definitions: List[str] = None,
        max_window: int = MAX_WINDOW,
        threshold: float = 0.8,
        similarity_measure: str = "jaccard"
    ):
        """
        Extracts keyword-like spans from input text using approximate matching via SimString.

        Args:
            database_dir (str): Directory where the SimString database is or will be stored.
            text_path (str): Text file containing keywords (one per line).
            list_definitions (List[str]): Additional keywords.
            max_window (int): Max span length.
            threshold (float): Similarity threshold.
            similarity_measure (str): One of "cosine", "jaccard", "dice", etc.
        """
        assert database_dir is not None or text_path is not None or list_definitions is not None

        self.max_window = max_window
        self.threshold = threshold
        self.database_dir = Path(database_dir or tempfile.mkdtemp())
        self.similarity_measure = similarity_measure

        if not (self.database_dir / "terms.simstring").exists():
            self.build_database(text_path, list_definitions)

        self.reader = simstring.reader(str(self.database_dir / "terms.simstring"))
        self.reader.measure = getattr(simstring, similarity_measure)
        self.reader.threshold = threshold

    def build_database(self, text_path: str = None, list_definitions: List[str] = None):
        """
        Builds the SimString database from text or list of terms.
        """
        with SimstringWriter(self.database_dir) as db:
            if text_path:
                with open(text_path, "r", encoding="utf-8") as f:
                    for line in f:
                        term = line.strip()
                        if term:
                            db.insert(term.lower())
            if list_definitions:
                for term in list_definitions:
                    term = term.strip()
                    if term:
                        db.insert(term.lower())

    def extract(self, text: str):
        """
        Extracts keyword-like spans from input text.

        Returns:
            List[dict]: Each dict contains:
                - span: spaCy span
                - match: matched string
                - similarity: dummy 1.0 (since current reader does not return score)
        """
        doc = nlp(text)
        spans = self.generate_valid_spans(doc)
        raw_matches = self.match_spans(spans)
        return self.resolve_overlaps(raw_matches)

    def generate_valid_spans(self, doc, window_size=None):
        """
        Generate candidate spans of up to `window_size` tokens.
        """
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
        """
        Heuristics to check if span is a valid keyword candidate.
        """
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
        """
        Query each candidate span against the simstring reader.
        """
        matches = []
        for span in spans:
            text = span.text.lower().strip()
            if not text:
                continue
            candidates = self.reader.retrieve(f"##{text}##")
            for c in candidates:
                matches.append({
                    "span": span,
                    "match": c.strip("#"),
                    "similarity": 1.0  # SimString reader doesn't provide score
                })
        return matches

    def resolve_overlaps(self, matches: List[dict]):
        """
        Selects best non-overlapping matches based on span length.
        """
        final_matches = []
        used_tokens = set()

        # Sort by span length (and dummy similarity)
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
        database_dir="preprocessing/simstring_db",
        text_path='preprocessing/strings.txt'
    )
    results = extractor.extract(input_text)

    print('Final matches:')
    for r in results:
        print(f"- Span: '{r['span']}' | Match: '{r['match']}'")

    print('Number of matches:', len(results))


if __name__ == '__main__':
    main()
