from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer
import os

#tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=os.environ.get('HF_TOKEN'), use_fast=False, add_bos_token=False, add_eos_token=False)

engine = InfiniGramEngine(index_dir='/export/home/cse170020/Riccardo_T/word_counts/infini_gram_llama_filter_all/', eos_token_id=tokenizer.eos_token_id)

print('Vocab size: ', tokenizer.vocab_size)

input_ids = tokenizer.encode('molto sophie mooren')
print(input_ids)
count = engine.count(input_ids=input_ids)
print('Resulting count: ', count)

input_ids = tokenizer.encode('oui')
print(input_ids)
count = engine.ntd(prompt_ids=input_ids)['result_by_token_id']
#print(count)

#for k in count.keys():
#    print(f'ID: {k} - str: {tokenizer._convert_id_to_token(k)} - prob: {count[k]['prob']}')
    
