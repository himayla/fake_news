import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords',quiet=True)
nltk.download("punkt", quiet=True)
from textblob import TextBlob
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
# from unidecode import unidecode
# import contractions
import re

# Load NLTK stopwords and Snowball stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
trash_words = ["WASHINGTON", "CNN", "REUTERS", "Reuters"]

def n_label(df, label):
    """ Returns number of examples with a specific label """
    return len(df[df["label"] == label])

def preprocess(raw_txt):
    """ Perform basic preprocessing, both for text-based and argumentation-based pipeline """
    txt = raw_txt.replace(r"(?P<url>[a-zA-Z]+:\/\/[^\s]+)", "")
    
    # Remove IPs
    txt = txt.replace(r"\.[0]*", "")

    return txt

def prep_argumentation_based(raw_txt):
    # Remove twitter handles and picture by
    txt = re.sub(r"([A-Za-z]+(?: [A-Za-z]+)*) \(@([A-Za-z0-9_]+)\) ([A-Za-z]+\s\d{1,2},\s\d{4})", "Tweet: ", raw_txt)
    
    # More twitter handles
    txt = re.sub(r"(\b[A-Z][a-z]+\b\s[A-Z]\.\s[A-Z][a-z]+(?:,\s[Jr\.]*)?\s*\(.*?\))", "Tweet: ", txt)

    txt = re.sub(r"(?P<date>(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},\s\d{4})", "", txt)

    txt = re.sub(r"\w+\s*-\s*\w+\s*\(@[\w]+\)", "", txt)

    # Remove twitter links
    txt = re.sub(r"\bpic\.twitter\.com\/\w+\b", "", txt)

    # Remove picture by
    txt = re.sub(r"(Photo by .+?\.)", "", txt)

    txt = re.sub(r'[\n  ]', " ", txt) 

    # Fix quotation things
    txt = re.sub(r'[``“\'\'”’`’\']', "'", txt)
    txt = re.sub(r'U.S.', "United States", txt)

    txt = re.sub(r'[\(\)]', "", txt)

    txt = re.sub(r'-', "", txt) 

    txt = re.sub(r'(?<=[a-z])\.(?=[A-Z])', r'. ', txt) 

    txt = re.sub(r'Gov.', "Governor", txt)
    txt = re.sub(r'UPDATE: ', "", txt)
    txt = re.sub(r'  ', " ", txt)
    txt = re.sub(r' — ', ", ", txt) 

    words = word_tokenize(txt)

    clean_text = []
    for word in words:
        word = re.sub(r'``', "'", word)
        word = re.sub(r"''", "'", word)
        word = re.sub(r'-', "", word) 
        word = re.sub(r'\(\)', "", word) 

        if word not in trash_words:
            clean_text.append(word)
    
    txt = "".join([" "+ i if not i.startswith("'") and i not in string.punctuation else i for i in clean_text]).strip()
    txt = re.sub(r'WASHINGTON Reuters ', '', txt)

    #     if word not in trashwords:
    #         clean_text.append(word)
    
    # for w in clean_text:
    #     for p in string.punctuation:
    #         if p not in [",",".","'"]:
    #             w = w.replace(p, "")

    # txt = ' '.join(clean_text)

    return txt

def prep_text_based(raw_text):    
    words = word_tokenize(raw_text)

    # Remove stop words 
    txt = [word for word in words if not word.lower() in stop_words]

    for w in txt:
        for p in string.punctuation:
            w = w.replace(p, "")

    # Remove suffices by stemming TOO RIGID
    txt = ' '.join([stemmer.stem(w) for w in txt])

    # Correct spelling of words # DOESNT WORK CORRECTLY
    # doc = TextBlob(txt)
    # corrected = doc.correct()
    
    return txt
