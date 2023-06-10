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
import spacy
import contextualSpellCheck

# If error: python -m spacy download en_core_web_sm

# Load NLTK stopwords and Snowball stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
nlp = spacy.load('en_core_web_sm')
contextualSpellCheck.add_to_pipe(nlp)

# Load English dictionary
dictionary = enchant.Broker().request_dict("en_US")

def load_replacements(file_path):
    with open(file_path, 'r') as f:
        patterns = [line.strip() for line in f if line.strip()]
    return patterns

def fix_punctuation(text):
    # Define the regular expression pattern
    pattern = r'(?<!\b[A-Z])\.(?!\.\s)'

    # Use the pattern to find matches in the text
    matches = re.finditer(pattern, text)

    # Iterate over the matches and add a space after each period
    fixed_text = ''
    prev_end = 0
    for match in matches:
        start = match.start()
        end = match.end()
        fixed_text += text[prev_end:start] + '. '
        prev_end = end

    fixed_text += text[prev_end:]

    return fixed_text

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

def replace_money_amounts(text):
    pattern = r"\$(\d+(\.\d{2})?)"
    result = re.sub(pattern, lambda match: format_money_amount(match.group(1)), text)
    return result

def format_money_amount(amount):
    amount = float(amount)
    if amount >= 1000000000:  # greater than or equal to 1 billion
        return f"over {amount / 1000000000:.2f} billion dollars"
    elif amount >= 1000000:  # greater than or equal to 1 million
        return f"over {amount / 1000000:.2f} million dollars"
    else:
        return f"over {amount:.2f} dollars"

def clean_text(raw_txt, dir):
    replacements = load_replacements(f'{dir}/replacements.txt')

    if 'text-based' in dir:
        words = word_tokenize(raw_txt)
        corrected_words = correct_spelling(words)
        words = [stemmer.stem(w) for w in corrected_words]
    else:
        for pattern in replacements:
            raw_txt = re.sub(pattern, '', raw_txt)
        
        raw_txt = re.sub(r'” ', ' ', raw_txt)
        raw_txt = re.sub(r"\’", "'", raw_txt)
        raw_txt = re.sub(r'’’', "", raw_txt)
        raw_txt = re.sub(r' ', ' ', raw_txt)
      
        raw_txt = re.sub(r"'s ", "s ", raw_txt)
        raw_txt = re.sub(r"We're ", "We are ", raw_txt)
        raw_txt = re.sub(r"n't ", " not", raw_txt)

        raw_txt = re.sub(r" weren t ", " were not ", raw_txt)
        raw_txt = re.sub(r" there's ", " there is ", raw_txt)
        raw_txt = re.sub(r" there s ", " there is ", raw_txt)
        raw_txt = re.sub(r" don t ", " do not ", raw_txt)
        raw_txt = re.sub(r"Don t ", "Do not ", raw_txt)

        raw_txt = re.sub(r"There s ", "There is ", raw_txt)
        raw_txt = re.sub(r"Weren t ", "Were not ", raw_txt)
        raw_txt = re.sub(r"won t ", "will not ", raw_txt)
        raw_txt = re.sub(r"Won t ", "Will not ", raw_txt)
        raw_txt = re.sub(r"\?\?\?", "?", raw_txt)

        raw_txt = re.sub(r'U.S.', "United States", raw_txt)
        raw_txt = re.sub(r'St. ', "Saint ", raw_txt)
        raw_txt = re.sub(r'Gen.', "General", raw_txt)
        raw_txt = re.sub(r' D.C.', " District County", raw_txt)

        raw_txt = re.sub(r'Dec.', "December", raw_txt)
        raw_txt = re.sub(r' & ', " and ", raw_txt)
        raw_txt = re.sub(r' J. ', " Junior ", raw_txt)

        raw_txt = re.sub(r' ll ', "", raw_txt)
        raw_txt = re.sub(r'diplomacy.While', "diplomacy. While", raw_txt)
        raw_txt = re.sub(r';', ":", raw_txt)
        raw_txt = re.sub(r', ', " ", raw_txt)
        raw_txt = re.sub(r'``', "", raw_txt)
        raw_txt = re.sub(r'\(', "", raw_txt)
        raw_txt = re.sub(r'\)', "", raw_txt)

        raw_txt = replace_money_amounts(raw_txt)
        raw_txt = fix_punctuation(raw_txt)
    
        words = word_tokenize(raw_txt)

        clean_words = []
        for word in words:
            new = re.sub(r'Oct.', 'October', word)
            new = re.sub(r'Gov. ', "Governor", new)
            new = re.sub(r'EU', "European Union", new)

            new = re.sub(r'Sen.', "Senator", new)
            new = re.sub(r"”’```''’’", "'", new)

            clean_words.append(new)
    
    clean_txt = "".join([" " + i if not i.startswith("'") and i not in [',','.','(',')','?',':','!','"','\'', "@", "$", '$'] else i for i in clean_words]).strip()


    doc = nlp(clean_txt)
    clean_txt = doc._.outcome_spellCheck

    return clean_txt