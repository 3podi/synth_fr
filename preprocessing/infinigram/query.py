from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)

engine = InfiniGramEngine(index_dir='/export/home/cse170020/Riccardo_T/word_counts/infini_gram_files/', eos_token_id=tokenizer.eos_token_id)

input_ids = tokenizer.encode('hospitalisation')
print(input_ids)
count = engine.count(input_ids=input_ids)
print(count)