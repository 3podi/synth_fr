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
    def __init__(self, database_path=None, text_path=None, max_window=5, threshold=0.7):
        assert database_path is not None or text_path is not None

        self.max_window = max_window
        self.threshold= threshold

        self.db = DictDatabase(CharacterNgramFeatureExtractor(3))
        if database_path is not None:
            self.db.load(database_path)
        else:
            self.build_database(text_path)
        
        self.searcher = Searcher(self.db, CosineMeasure())

    
    def build_database(self, path=None):
        # Read your file and add each line
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                string = line.strip()
                if string:  # skip empty lines
                    self.db.add(string.lower())

    def extract(self, text):
        doc = self.preprocess(text)
        spans = self.generate_valid_spans(doc, self.max_window)
        raw_matches = self.match_spans_with_simstring(spans, self.searcher, self.threshold)
        final_matches = self.resolve_overlaps(raw_matches)
        return final_matches
    
    def preprocess(self, text):
        return nlp(text)
    
    def is_valid_sequence(self, span):
        tokens = list(span)
        if len(tokens) == 0 or len(tokens) > MAX_WINDOW:
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
        valid_spans = []
        for sent in doc.sents:
            for i in range(len(sent)):
                for j in range(i, min(i + window_size, len(sent))):
                    span = sent[i:j + 1]
                    if self.is_valid_sequence(span):
                        valid_spans.append(span)
        return valid_spans

    def match_spans_with_simstring(self, spans, searcher, threshold=0.7):
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