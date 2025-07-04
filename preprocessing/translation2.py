from datasets import load_dataset
from datasets import Dataset
import pandas as pd

from transformers import pipeline
import nltk

nltk.download("punkt")

def chunk_sentences(sentences, chunk_size=1, overlap=0):
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + chunk_size
        chunks.append(" ".join(sentences[start:end]))
        start += chunk_size - overlap
    return chunks


def split_sentences(examples):
    all_sentences = []
    ids = []
    for i, text in enumerate(examples["text"]):
        sents = nltk.sent_tokenize(text)
        chunks = chunk_sentences(sents)
        all_sentences.extend(chunks)
        ids.extend([i] * len(chunks))
    return {"sentence": all_sentences, "orig_id": ids}

def translate_batch(examples, translator=None):
    translations = translator(examples["sentence"], truncation=True)
    return {"translation": [t["translation_text"] for t in translations]}

def main():
    
    dataset = load_dataset("parquet", data_files="../data/mimic.parquet")  # yields a DatasetDict or Dataset
    sent_dataset = dataset.map(split_sentences, batched=True, remove_columns=dataset['train'].column_names)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en", max_length=512, device=0)
    translator = pipeline("translation_en_to_fr", device=0)
    
    print(translator.model.device)  # Should say 'cuda:0' if on GPU

    # Translate sentence chunks
    translated_dataset = sent_dataset.map(
        lambda x: translate_batch(x, translator=translator),
        batched=True,
        batch_size=16,
        remove_columns=["sentence"]
    )

    
    # Recombine translated chunks into full documents
    df = translated_dataset['train'].to_pandas()
    grouped = df.groupby("orig_id")["translation"].apply(lambda chunks: " ".join(chunks)).reset_index()
    result_dataset = Dataset.from_pandas(grouped.rename(columns={"translation": "translated_text"}))

    # Show a sample
    print(result_dataset[0])
    result_dataset.to_parquet('../data/sft_sample/mimic_translate.parquet')

    
if __name__ == "__main__":
    main()
    #long_text = (
#            "A 53-year-old Asian woman comes to the physician because of a 2-month history of severe pain in her right leg while walking. She used to be able to walk a half-mile (800-m) to the grocery store but has been unable to walk 200 meters without stopping because of the pain over the past month. She can continue to walk after a break of around 5 minutes. She has hypertension, atrial fibrillation, and type 2 diabetes mellitus. She has smoked one pack of cigarettes daily for the past 32 years. Current medications include metformin, enalapril, aspirin, and warfarin. Vital signs are within normal limits. Examination shows an irregularly irregular pulse. The right lower extremity is cooler than the left lower extremity. The skin over the right leg appears shiny and dry. Femoral pulses are palpated bilaterally; pedal pulses are diminished on the right side."
#    )

#    translated_text = translate_long_text(long_text, translator, chunk_size=2, overlap=0) #Overlap fails because different contexts give different translations so its impossible to join the overlapped chunks of translated text
#    print("Translated Text:\n", translated_text)


