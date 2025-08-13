import os
import pandas as pd
import argparse
import unicodedata
import re
import string
import csv
import pickle
import numpy as np
from collections import Counter

def remove_symbols(text):
    # Remove everything that's not a letter, digit, or whitespace
    return re.sub(r'[^\w\s]', '', text)

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