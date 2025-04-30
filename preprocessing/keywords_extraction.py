from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.cosine import CosineMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher

import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    return nlp(text)

VALID_POS_START_END = {"CONJ", "ADP", "DET"}
MAX_WINDOW = 5

class KeywordsExtractor:
    def __init__(self, database_path=None, text_path=None, list_definitions=None, max_window=MAX_WINDOW, threshold=0.8):
        """
        Extracts keywords from text using n-gram similarity matching against a reference database.
        
        This class uses a SimString-based database with cosine similarity to find approximate 
        matches of keyword-like phrases in a given input text. It uses spaCy for NLP preprocessing 
        and filters valid spans based on part-of-speech tags and token rules.

        Args:
            database_path (str, optional): Path to an existing SimString database file.
            text_path (str, optional): Path to a text file to build the keyword database from.
            max_window (int, optional): Maximum number of tokens in a keyword span (default: 5).
            threshold (float, optional): Cosine similarity threshold for matching (default: 0.7).

        Raises:
            AssertionError: If neither `database_path` nor `text_path` is provided.
        """
        assert database_path is not None or text_path is not None or list_definitions is not None

        self.max_window = max_window
        self.threshold= threshold

        self.db = DictDatabase(CharacterNgramFeatureExtractor(3))
        if database_path is not None:
            self.db.load(database_path)
        else:
            self.build_database(text_path, list_definitions)
        
        self.searcher = Searcher(self.db, CosineMeasure())

    
    def build_database(self, text_path=None, list_definitions=None):
        """
        Builds a SimString database from a newline-separated text file.

        Args:
            text_path (str): Path to the text file containing keywords, one per line.
            list_definitions (List, str): list of additional keywords
        """

        # Read your file and add each line
        if text_path:
            with open(text_path, "r", encoding="utf-8") as f:
                for line in f:
                    string = line.strip()
                    if string:  # skip empty lines
                        self.db.add(string.lower())
        
        if list_definitions:
            for definition in list_definitions:
                definition = definition.strip()
                if definition:
                    self.db.add(definition.lower())

    def extract(self, text):
        """
        Extracts and matches keyword spans from the input text.

        Args:
            text (str): Input text to analyze.

        Returns:
            List[dict]: List of dictionaries with matched span info:
                {
                    'span': spaCy span,
                    'match': matched string from DB,
                    'similarity': float score
                }
        """

        doc = self.preprocess(text)
        spans = self.generate_valid_spans(doc, self.max_window)
        raw_matches = self.match_spans_with_simstring(spans, self.searcher, self.threshold)
        final_matches = self.resolve_overlaps(raw_matches)
        return final_matches
    
    def preprocess(self, text):
        """
        Applies NLP preprocessing to the text using spaCy.

        Args:
            text (str): Raw input string.

        Returns:
            spaCy Doc: Tokenized and parsed document.
        """
        return nlp(text)
    
    def is_valid_sequence(self, span):
        """
        Checks if a span is a valid keyword candidate.

        Args:
            span (spaCy Span): Span of tokens to validate.

        Returns:
            bool: True if the span is valid for matching, False otherwise.
        """

        tokens = list(span)
        if len(tokens) == 0 or len(tokens) > self.max_window:
            return False
        if span.start == span.end:  # single-token span
            tok = tokens[0]
            if tok.is_stop or tok.like_num:
                return False
        if tokens[0].pos_ in VALID_POS_START_END or tokens[-1].pos_ in VALID_POS_START_END:
            return False
        if tokens[0].is_punct:
            return False
        return True
    
    def generate_valid_spans(self, doc, window_size=MAX_WINDOW):
        """
        Generates valid keyword spans from the text based on a sliding window.

        Args:
            doc (spaCy Doc): Preprocessed text.
            window_size (int): Max number of tokens to consider for a span.

        Returns:
            List[spaCy Span]: Valid spans to match against the database.
        """

        valid_spans = []
        for sent in doc.sents:
            for i in range(len(sent)):
                for j in range(i, min(i + window_size, len(sent))):
                    span = sent[i:j + 1]
                    if self.is_valid_sequence(span):
                        valid_spans.append(span)
        return valid_spans

    def match_spans_with_simstring(self, spans, searcher, threshold=0.7):
        """
        Finds approximate matches for each span using SimString.

        Args:
            spans (List[spaCy Span]): Spans to match.
            searcher (SimString Searcher): Searcher object to query the DB.
            threshold (float): Cosine similarity threshold.

        Returns:
            List[dict]: Match information for each span.
        """

        matches = []

        for span in spans:
            span_text = span.text.lower().strip()
            if not span_text:
                continue
            results = searcher.ranked_search(span_text, threshold)
            for r in results:
                matches.append({
                    "span": span,
                    "match": r,
                    "similarity": results[r]
                })
        return matches
    
    def resolve_overlaps(self, matches):
        """
        Removes overlapping matches (one token/s can only be matched with one string in the db),
        keeping the best match based on similarity and length.

        Args:
            matches (List[dict]): Raw matches from SimString.

        Returns:
            List[dict]: Non-overlapping filtered matches.
        """

        final_matches = []
        used_tokens = set()

        # Sort by similarity (or span length if similarity is same)
        matches = sorted(matches, key=lambda x: (-x["similarity"], -(x["span"].end - x["span"].start)))

        for match in matches:
            token_idxs = set(range(match["span"].start, match["span"].end))
            if used_tokens.intersection(token_idxs):
                continue  # overlaps with already used
            final_matches.append(match)
            used_tokens.update(token_idxs)

        return final_matches


def main():

    input_text = 'The patient suffer of Vertigo and nausea and Migraine and Lantus (Insulin Glargine).'

    extractor = KeywordsExtractor(text_path='preprocessing/strings.txt')
    results = extractor.extract(input_text)

    print('Final matches: ', results)
    print('Len final matches: ', len(results))

if __name__ =='__main__':
    main()