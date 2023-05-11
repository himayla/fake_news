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
from unidecode import unidecode
import contractions

# Load NLTK stopwords and Snowball stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def n_label(df, label):
    """ Returns number of examples with a specific label """
    return len(df[df["label"] == label])

def preprocess(raw_text):
    """ Perform preprocessing"""

    # Remove urls and IPs
    txt = raw_text.replace(r"http://\S+|https://\S++", "").replace(r"\.[0]*", "")

    words = word_tokenize(unidecode(txt))

    for i in range(len(words)):
        for p in string.punctuation:
            words[i] = words[i].replace(p, "")
    
    # Remove stop words 
    txt = [word for word in words if not word.lower() in stop_words]

    # Remove suffices by stemming TOO RIGID
    #txt = ' '.join([stemmer.stem(w) for w in txt])

    txt = " ".join(txt)

    expanded_text = []
    for word in clean_text.split():
        expanded_text.append(contractions.fix(word))  
    clean_text = ' '.join(expanded_text)
    clean_text = re.sub(r'[``“”’,"\'-]', "", clean_text) # Remove weird quotation things
    clean_text = re.sub(r'\n', " ", clean_text) # Remove weird quotation things
    txt = txt.replace("  ", " ")

    # # Correct spelling of words # DOESNT WORK CORRECTLY
    # doc = TextBlob(txt)
    # corrected = doc.correct()

    return txt