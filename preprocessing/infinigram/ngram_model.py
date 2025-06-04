import os
from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer
from tqdm import tqdm
import math
from collections import Counter
import pickle
import argparse

class NGramModel:
    def __init__(self, 
                 index_folder_path,
                 save_path,
                 max_len,
                 beam_size,
                 top_k,
                 lower_limit_count,
                 upper_limit_count,
                 vocab_path=None):
        
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=os.environ.get('HF_TOKEN'), use_fast=False, add_bos_token=False, add_eos_token=False)
        self.engine = InfiniGramEngine(index_dir=index_folder_path, eos_token_id=self.tokenizer.eos_token_id)

        # Beam search params
        self.max_len=max_len
        self.beam_size=beam_size
        self.top_k=top_k
        self.lower_limit_count=lower_limit_count
        self.upper_limit_count=upper_limit_count

        # Add vocabulary for vab based generation completion
        if vocab_path:
            with open(vocab_path, 'rb') as f:
                word_counts = pickle.load(f)
            
            # Get all counts and sort them
            if isinstance(word_counts, Counter):
                self.vocab = set([word for word, count in word_counts])
            else:
                self.vocab = list(word_counts.values())
        
        self.save_path = save_path

    def beam_search(self):
        
        beam= self.initialize_beam_no_subwords()

        for length in range(1, self.max_len + 1):
            print('Len: ', length)
            new_beam = []

            for seq, log_p in tqdm(beam):
                
                top_next = self.next_token_distribution(seq)

                for token, prob in top_next:
                    new_seq = seq + [token]
                    new_logp = log_p + math.log(prob)
                    if length==1 and new_logp == 0.0:
                        continue
                    count = self.engine.count(input_ids=new_seq)
                    if count['count']>self.upper_limit_count or count['count']<self.lower_limit_count:
                        continue
                    new_beam.append((new_seq, new_logp))

            # Prune to top beam_size
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_size]
            
            #if self.vocab:
            #    beam = self.complete_unfinished_words(beam)

            self.decode_and_save(self.save_beam_path, length)


    def initialize_beam_no_subwords(self):
        
        vocab = self.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        
        # Filter to get only full-word tokens (not subwords)
        full_word_token_ids = [
            token_id for token_id, token_str in inv_vocab.items()
            if token_str.startswith('▁')  # SentencePiece: full word tokens
        ]
        
        # Initialize the beam with full-word tokens
        return [([token_id], 0.0) for token_id in full_word_token_ids]
    
    def next_token_distribution(self, seq):

        token_results = self.engine(prompt_ids=seq)['result_by_token_id']
        sorted_probs = sorted(
            ((int(k), v['prob']) for k, v in token_results.items()),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_probs[:self.top_k]
    
    def detokenize(self, ids_seq):
        tok_seq = [self.tokenizer._convert_id_to_token(idx) for idx in ids_seq]
        return self.tokenizer.convert_tokens_to_string(tok_seq)
    
    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def complete_unfinished_words(self, text_list):
        """
        For each beam entry, if the last word is not complete (not in vocab),
        use the n-gram model to extend it token-by-token until it becomes a known word.
        """
        completed_texts = []

        for text in text_list:
            words = text.split()

            if not words:
                completed_texts.append(text)
                continue

            last_word = words[-1]

            if last_word in self.vocab:
                completed_texts.append(text)
                continue

            # Iteratively complete the last word using n-gram model
            context_tokens = self.tokenize(text)
            while last_word not in self.vocab:
                next_token_probs = self.next_token_distribution(context_tokens)

                if not next_token_probs:
                    # No suggestions, stop
                    break

                # Pick the highest-probability token
                next_token = next_token_probs[0][0]

                context_tokens.append(next_token)
                text = self.detokenize(context_tokens)
                words = text.split()
                last_word = words[-1] if words else ''

            completed_texts.append(text)

        return completed_texts
    
    def postprocess_with_completion(self, file_path):
        """
        Reads a file of beam search outputs (log_p \t text),
        applies n-gram word completion, and returns the completed texts.

        Args:
            file_path (str): Path to a .tsv output file.

        Returns:
            list[str]: Completed text strings.
        """
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    _, text = parts
                    texts.append(text)
        
        completed_texts = self.n_gram_model.complete_unfinished_words(texts)
        return completed_texts
    
    def decode_and_save(self, seqs, n):
        """
        Decodes a list of token ID sequences and saves the resulting text lines to a file.
        """

        os.makedirs(self.save_path, exist_ok=True)  # Ensure the directory exists

        with open(f'{self.save_path}/tokens_{n}_beam_size_{self.beam_size}.tsv', 'w', encoding='utf-8') as f:
            for ids_seq, log_p in seqs:
                text = self.detokenize(ids_seq).strip()
                f.write(f"{log_p:.6f}\t{text}\n")

def save_completed_texts(completed_texts, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in completed_texts:
            f.write(text.strip() + '\n')

    
def main(args):
    
    n_gram_model = NGramModel(
                    index_folder_path=args.index_folder,
                    save_path=args.save_path,
                    max_len=args.n,
                    beam_size=args.beam_size,
                    top_k=args.top_k,
                    lower_limit_count=100,
                    upper_limit_count=1000,
                    vocab_path=args.vocab_path   
                    )
    
    n_gram_model.beam_search()

    files = os.listdir(args.save_path)
    files = [f for f in files if f.startswith('tokens')]

    for f in files:
        file_path = os.path.join(args.save_path,f)
        completed_texts = n_gram_model.postprocess_with_completion(file_path)
        save_completed_texts(completed_texts, os.path.join(args.save_path,f'completed_{f}'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beam search to find the most common n-grams.")
    parser.add_argument("index_folder", help="Path to folder with the infinigram indexing")
    parser.add_argument("save_path", help="Path to folder where to save retrivied expression")
    parser.add_argument("output_folder", help="Path to output n-grams")
    parser.add_argument("n", type=int, help="N-grams size")
    parser.add_argument("--beam_size", type=int, default=1000, help="Size of the beam")
    parser.add_argument("--top_k", type=int, default=20, help="Top k results to keep for each next token distribution")
    parser.add_argument("vocab_path", type=str, default=None, help="Path to vocab")

    args = parser.parse_args()

    main(args)