import enchant
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords',quiet=True)
nltk.download("punkt", quiet=True)
import pandas as pd
import re
import string

# Load NLTK stopwords and Snowball stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# Load English dictionary
dictionary = enchant.Broker().request_dict("en_US")

def load_replacements(file_path):
    with open(file_path, 'r') as f:
        patterns = [line.strip() for line in f if line.strip()]
    return patterns

def correct_spelling(words):
    corrected_words = []
    for word in words:
        if not dictionary.check(word):
            suggestions = dictionary.suggest(word)
            if suggestions:
                corrected_word = suggestions[0]
            else:
                corrected_word = word  
            corrected_words.append(corrected_word)
        else:
            corrected_words.append(word)
    return corrected_words

def clean_text(raw_txt, dir):
    replacements = load_replacements(f'{dir}/replacements.txt')
    for pattern in replacements:
        txt = re.sub(pattern, '', raw_txt)

    words = word_tokenize(txt)

    if 'text-based' in dir:
        corrected_words = correct_spelling(words)
        words = [stemmer.stem(w) for w in corrected_words]
    else:
        for word in words:
            word = re.sub(r'U.S.', "United States", word)
            word = re.sub(r'Gov.', "Governor", word)
            word = re.sub(r'Sen.', "Senator", word)
            word = re.sub(r"|”|’`|``|''", "''", word)

    txt = "".join([" " + i if not i.startswith("'") and i not in [',','.','(',')'] else i for i in words]).strip()

    return txt