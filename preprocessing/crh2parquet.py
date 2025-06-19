import os
import pandas as pd
from docx import Document
import argparse
import re

OUTPUT_PATH = (
    f"datasets/health/"
)
PROMPT_PATH = "datasets/preprocessing/health/prompt.txt"

SPLIT_STRINGS = ['Compte rendu d’hospitalisation', 'COMPTE-RENDU ANATOMO PATHOLOGIQUE']
REPLACE_STRINGS = ['Diagnostic principal motivant l’hospitalisation (code CIM 10) : ', 'Acte principal : ', 'Acte CCAM principal  : ']

def check_icd_code(s):
    pattern = r'\[[A-Z]+\d+\]'
    return bool(re.search(pattern, s))

def split_text_and_bracketed(input_string):
    match = re.match(r'^(.*?)\s*\[(.*?)\]\s*$', input_string)
    if match:
        return match.group(1), match.group(2)
    
def split_text_and_bracketed2(input_string):
    match = re.search(r'^(.*?)\s*\[(.*?)\]', input_string)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    #else:
    #    return input_string.strip(), None
    
def split_text_and_bracketed3(input_string):
    # Match the last occurrence of uppercase letters inside square brackets
    match = re.search(r'^(.*?)\s*\[([A-Z]+)\](?!.*\[[A-Z]+\])$', input_string)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    else:
        return input_string.strip(), None
    
def extract_squared_bracket_content(text):
    # Regular expression to find content within square brackets containing only uppercase letters or digits
    pattern = r'\[([A-Z0-9]+)\]'
    matches = re.findall(pattern, text)
    return matches

def remove_square_brackets_content(text):
    # Regular expression to match content within square brackets
    pattern = r'\[.*?\]'
    # Replace matched content with an empty string
    return re.sub(pattern, '', text)

def remove_duplicates(list1, list2):
    seen = set()
    result1, result2 = [], []
    for item1, item2 in zip(list1, list2):
        if item1 not in seen:
            seen.add(item1)
            result1.append(item1)
            result2.append(item2)
    return result1, result2
    
def normalize_whitespace(text):
    """
    Replaces multiple consecutive whitespace characters with a single space,
    and strips leading/trailing whitespace.

    Args:
        text (str): Input string.

    Returns:
        str: Cleaned string with normalized whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()
    
class DataProcessorDocx:
    def __init__(self, docs_folder_path):
        
        self.docx_files = self.get_docx_paths(docs_folder_path)
        print('Docx read: ', len(self.docx_files))
        
    def get_docx_paths(self, root_folder):
        docx_files = []

        for current_dir, dirs, files in os.walk(root_folder):
            # If there are no subdirectories, it's a leaf directory
            if not dirs:
                # Look for .docx files in this leaf directory
                for file in files:
                    if file.lower().endswith('.docx'):
                        docx_files.append(os.path.join(current_dir, file))

        return docx_files
    
    def extract_text_from_docx(self, file_path):
        """
        Extracts all text from a .docx file.

        Args:
            file_path (str): Path to the .docx file.

        Returns:
            str: Extracted text from the document.
        """
        doc = Document(file_path)
        full_text = []

        # Extract text from paragraphs
        for para in doc.paragraphs:
            full_text.append(para.text)
        
        return full_text
    
    
    def process_data(self):
        count_crh = 0
        count_resume = 0
        for path in self.docx_files:
            found_resume = False
            text = self.extract_text_from_docx(path)
            #print(text)
            for t in text:
                
                if check_icd_code(t):
                    t = t.replace('Diagnostic principal motivant l’hospitalisation (code CIM 10) : ', '')
                    #print('FOUND CODE: ', t)
                
                if 'Compte rendu d’hospitalisation' in t or 'COMPTE-RENDU ANATOMO PATHOLOGIQUE' in t:
                    count_crh = count_crh + 1
                    found_crh = True
                
                if 'Résumé structuré' in t:
                    count_resume = count_resume + 1
                    found_resume = True
            if not found_resume:
                print('\nNO RESUME: ', text)
            #print(text)
        
        print('crh count: ', count_crh)
        print('count_resume: ', count_resume)
        
    def process_data2(self):
        
        all_texts = []
        all_codes = []
        all_definitions = []
        
        for path in self.docx_files:
            crh_pos = -1
            resume_pos = -1
            codes = []
            definitions = []
            
            text = self.extract_text_from_docx(path)
            for i, t in enumerate(text):
                
                if check_icd_code(t):                
                    for replace_string in REPLACE_STRINGS:
                        t = t.replace(replace_string,'')
                        
                    matched_brakets = extract_squared_bracket_content(t)
                    code = matched_brakets[-1]
                    codes.append(code)
                    
                    definition = remove_square_brackets_content(t)
                    definitions.append(normalize_whitespace(definition))
                    #print(definitions[-1], code)
                    #if code == 'permanente':
                    #    print(text)
                    #codes.append(code)
                
                for string in SPLIT_STRINGS:
                    if string in t:
                        crh_pos = i
                
                if 'Résumé structuré' in t:
                    resume_pos = i
            
            assert len(codes) == len(definitions)
            
            crh_text = normalize_whitespace(" ".join(text[crh_pos+1:resume_pos]))
            codes, definitions = remove_duplicates(codes, definitions)
            #print(codes)
            codes = " ".join(codes)
            definitions = "###".join(definitions)
            
            all_texts.append(crh_text)
            all_codes.append(codes)
            all_definitions.append(definitions)
            
            print(crh_text)
            #print('\n')
            #print(codes)
            #print('\n')
            #print(definitions)
            #print('\n')
            
            #if crh_pos == -1 or resume_pos == -1:
            #    print('TEXT: ', text)
            #    print('\n')
            #    print('EXTRACTION: ', crh_text)
            #if crh_pos != -1 and resume_pos != -1:
            #    crh_text = " ".join(text[crh_pos+1:resume_pos])
            #elif crh_pos != -1 and resume_pos == -1:
            #    crh_text = " ".join(text[crh_pos+1:])
        
        print(len(all_texts), len(all_codes), len(all_definitions))
        return all_texts, all_codes, all_definitions    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description= 'Setting note path')
    parser.add_argument('docs_folder_path', type=str, help='Path to the folder cotaining all the docx')    
    parser.add_argument('output_path', type=str, help='Path output_file')
    args = parser.parse_args()

    processor = DataProcessorDocx(args.docs_folder_path)
    texts, codes, definitions = processor.process_data2()
    
    df = pd.DataFrame(list(zip(texts, definitions, codes)), columns=['text', 'keywords', 'code'])
    df.to_parquet(args.output_path)
