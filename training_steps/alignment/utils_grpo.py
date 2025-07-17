from collections import Counter
from typing import List
import unicodedata
import re
import string

import torch
from transformers import AutoTokenizer
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
import pandas as pd

reasoning_start = "<réfléchir>"
reasoning_end = "</réfléchir>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

SYSTEM_PROMPT = f"""Vous êtes un médecin hautement spécialisé. Votre tâche consiste à rédiger un rapport d'hospitalisation fictif complet et réaliste. Le document doit présenter une évolution clinique cohérente avec une terminologie médicale précise, tout en respectant scrupuleusement la séquence imposée des mots-clés. Retournez uniquement le rapport entre {reasoning_start} et {reasoning_end}. Donnez ensuite la séquence des codes icd-10 associés à le rapport sous la forme d'une séquence de nombres entiers et placez-les entre  {solution_start} et  {solution_end}."""

SYSTEM_PROMPT2 = f"""Vous êtes un médecin hautement spécialisé. Votre tâche consiste à rédiger un rapport d'hospitalisation fictif complet et réaliste. Le document doit présenter une évolution clinique cohérente avec une terminologie médicale précise, tout en respectant scrupuleusement la séquence imposée des mots-clés. """

def remove_accents(text):
    """Remove accents and special characters from Unicode text."""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

def normalize_text(text):

    # Normalize common Unicode dashes to hyphen
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')   

    # Replace hyphens with spaces
    text = text.replace('-', ' ')
    
    # Remove accents
    text = remove_accents(text)
    
    # Remove invisible/non-printable characters
    text = ''.join(c for c in text if c.isprintable())
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Replace all punctuation with whitespace
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # Lowercase
    text = text.lower()
    
    return re.sub(r'\s+', ' ', text).strip()

def format_grpo(example):
    
    return {
        "prompt" : [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": example["keywords"]},
        ],
        "answer": example["solution"],
    }

def format_grpo2(example):
    
    return {
        "prompt" : [
        {"role": "system", "content": SYSTEM_PROMPT2},
        {"role": "user",   "content": example["keywords"]},
        ],
        "answer": "",
    }

# Optional EOS token after </SOLUTION>
#solution_end_regex = r"</SOLUTION>\s*(?:" + re.escape(tokenizer.eos_token) + ")?"
solution_end_regex = r"</SOLUTION>"

# Match only whitespace-separated integers between <SOLUTION> and </SOLUTION>
integer_seq_pattern = r"(?:\d+\s*)+"

match_format = re.compile(
    rf"{reasoning_end}.*?"                        # Match reasoning end to solution start
    rf"{solution_start}"                          # Match <SOLUTION>
    rf"({integer_seq_pattern})"                   # Capture only integer sequence
    rf"{solution_end_regex}",                      # Match </SOLUTION> + optional EOS
    #rf"\s*$",                                     # Optional trailing whitespace till end of string
    flags=re.MULTILINE | re.DOTALL
)

match_text = re.compile(
    re.escape(reasoning_start) + r"(.*?)" + re.escape(reasoning_end),
    re.DOTALL
)

def reward_match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

def count_repeats_penalty(pred_codes: List[str]):
    counts = Counter(pred_codes)
    penalty = sum(1 for freq in counts.values() if freq > 1)
    reward = -1 * penalty/len(counts.values()) 
    return reward

def compute_f1(pred_codes: List[str], ref_codes: List[str]):
    common = set(pred_codes) & set(ref_codes)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_codes)
    recall = num_same / len(ref_codes)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def reward_f1(completions, answer, **kwargs):
    
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]
    
    scores = []   
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        scores.append(compute_f1(guess.split(), true_answer.split()))
    
    return scores

def reward_no_repeat(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]
    
    scores = []   
    for guess in extracted_responses:
        if guess is None:
            scores.append(-1)
            continue
        
        scores.append(count_repeats_penalty(guess.split()))
    
    return scores

### CAREFULL: the keywords must be not normalized -> redo processing (or normalize text) (carefull partages keywords are not normalized)
def reward_matching_keywords(prompts, completions, **kwargs):
    scores = []

    for prompt, response_group in zip(prompts, completions):
        question = prompt[-1]["content"]  # Get the prompt for this sample
        responses = [completion["content"] for completion in response_group]

        prompt_keywords = question.split(",")
        prompt_keywords = [kw.strip() for kw in prompt_keywords]
        print("Found those prompt keywords: ", prompt_keywords)

        extracted_texts = [
            guess.group(1)
            if (guess := match_text.search(r)) is not None else None
            for r in responses
        ]

        for text in extracted_texts:
            if text is None:
                #scores.append(len(prompt_keywords) * -1)
                scores.append(-1)
            else:
                scores.append(sum(1 for key in prompt_keywords if normalize_text(key) in normalize_text(text))/len(prompt_keywords))

    return scores

def reward_matching_keywords2(prompts, completions, **kwargs):
    scores = []
    
    print('CO?PLETIONS: ', completions)

    for prompt, response_group in zip(prompts, completions):
        question = prompt[-1]["content"]  # Get the prompt for this sample
        responses = [completion["content"] for completion in response_group]

        prompt_keywords = question.split(",")
        prompt_keywords = [kw.strip() for kw in prompt_keywords]
        print("Found those prompt keywords: ", prompt_keywords)

        for text in responses:
            if text is None:
                #scores.append(len(prompt_keywords) * -1)
                scores.append(-1)
            else:
                scores.append(sum(1 for key in prompt_keywords if normalize_text(key) in normalize_text(text))/len(prompt_keywords))

    return scores

### TODO: balance right completion score and right answer score, getting the right answer is a lot more diffucult so should deserve a lot higher score
### TODO: answer length reward, do i need to tokenize again? 

class ScoringModelRewardFunction:
    def __init__(self, sts_model, reference_data_path=None):
        self.sts_model = sts_model
        self.sts_model.eval()
        for param in self.sts_model.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.sts_model._first_module().auto_model.config._name_or_path)
        self.reference_data = pd.read_parquet(f"datasets/health/grpo/public_reference_grpo.parquet")
        self.reference_data = self.reference_data["response"].tolist()

    def __call__(self, completions, **kwargs):
        # Use the model to compute something
        scores = []
        responses = [completion[0]["content"] for completion in completions]
        scores.extend(self.compute_similarities(responses))
        return scores    
    
    def compute_similarities(self, generated_data):
        
        all_chunks_ref = []

        for idx, text in enumerate(self.reference_data):
            chunks = self.split_into_chunks(text, max_length=514)
            all_chunks_ref.extend(chunks)
            
        all_chunks = []
        chunk_indices = []  # Track which chunks belong to which text

        for idx, text in enumerate(generated_data):
            chunks = self.split_into_chunks(text, max_length=514)
            all_chunks.extend(chunks)
            chunk_indices.extend([idx] * len(chunks))

        with torch.no_grad():
            embeddings1 = self.sts_model.encode(all_chunks_ref, batch_size=32, convert_to_tensor=True)
            embeddings2 = self.sts_model.encode(all_chunks, batch_size=32, convert_to_tensor=True)
                

        final_embeddings = []
        num_texts = len(generated_data)

        for idx in range(num_texts):
            # Select embeddings for current text
            mask = torch.tensor([i == idx for i in chunk_indices])
            text_chunk_embeddings = embeddings2[mask]

            # Aggregate embeddings (mean pooling)
            aggregated_embedding = text_chunk_embeddings.mean(dim=0)
            final_embeddings.append(aggregated_embedding)
            
        final_embeddings = torch.stack(final_embeddings)  # Shape: [num_texts, embedding_dim]

        # shape: [len(reference_data), len(generated_data)]
        similarity_matrix = cos_sim(embeddings1, final_embeddings)

        # Average across reference data for each generated sample:
        mean_scores_per_generated = similarity_matrix.mean(dim=0)
        
        return mean_scores_per_generated.tolist()  # Return as list of floats
    
    def split_into_chunks(self, text, max_length=514):
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i:i+max_length]
            chunks.append(self.tokenizer.convert_tokens_to_string(chunk))
        return chunks