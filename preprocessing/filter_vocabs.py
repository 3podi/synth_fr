from keywords_extraction import KeywordsExtractor
import argparse
import pickle
import os
from tqdm import tqdm

def main(args):

    texts = []
    with open(args.vocab_path2, 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip().split('\t')[0])

    with open(args.vocab_path1, 'rb') as file:
        definitions = pickle.load(file)
    definitions = definitions.keys()

    extractor = KeywordsExtractor(text_path=None, list_definitions=definitions, threshold=args.threshold)
    filtered_expressions = []

    for text in tqdm(texts):
        candidates = extractor.reader.retrieve(text)        
        if len(candidates) > 0:
            filtered_expressions.append(text)
        
    os.makedirs(args.save_path, exist_ok=True)  # Ensure the directory exists
    file_name = args.vocab_path2.split('/')[-1]    
    with open(f'{args.save_path}/filtered_{file_name}', 'w', encoding='utf-8') as f:
        for text in filtered_expressions:
            f.write(text.strip() + '\n')
    
    if args.print:
        print('Number of filtered expressions: ', len(texts)-len(filtered_expressions))
        print('Remaining expressions: ', filtered_expressions)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Set paths to vocabs to filter the second from the entries of the first")
    parser.add_argument("vocab_path1", help="Path to vocab_1")
    parser.add_argument("vocab_path2", help="Path to vocab_2 (must be a .tsv for now)")
    parser.add_argument("save_path", type=str, help="Folder where to save output file")
    parser.add_argument("--threshold", type=int, default=0.8, help="Similarity threshold")
    parser.add_argument("--print", action='store_true', help="Optionally print filtered ")

    args = parser.parse_args()

    assert args.vocab_path2.endswith('.tsv')

    main(args)
