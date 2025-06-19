from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

#AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
#AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

#AutoModelForCausalLM.from_pretrained("meta-llama/llama-2-7b-hf")
#AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-hf")

#AutoModelForCausalLM.from_pretrained("xz97/AlpaCare-llama2-13b", use_safetensors=True)
#AutoTokenizer.from_pretrained("xz97/AlpaCare-llama2-13b")

#SentenceTransformer("FremyCompany/BioLORD-2023")
#AutoTokenizer.from_pretrained("FremyCompany/BioLORD-2023")

SentenceTransformer("FremyCompany/BioLORD-2023-M")
AutoTokenizer.from_pretrained("FremyCompany/BioLORD-2023-M")

#AutoModelForCausalLM.from_pretrained("google/medgemma-4b-it")
#AutoTokenizer.from_pretrained("google/medgemma-4b-it")

print("DOWNLOADED!")
