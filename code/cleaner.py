import contextualSpellCheck
import enchant
import nltk
from nltk.corpus import stopwords; nltk.download('stopwords',quiet=True)
from nltk.stem.snowball import SnowballStemmer; nltk.download("punkt", quiet=True)
from nltk.tokenize import word_tokenize
import re 
import spacy # If error: python -m spacy download en_core_web_sm

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
    pattern = r'\b[A-Za-z]+\.[A-Za-z]+\b(?!\.)'

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
    replaced_text = re.sub(r"\$([\d,.]+)", replace_match, text)

    # print(replaced_text)

    return replaced_text


def replace_match(match):
    amount = match.group(1)
    if '.' in amount:
        amount = 'over ' + amount.split('.')[0]
    else:
        amount = amount.replace(',', '') + ' dollars'
    return amount

def clean_text(raw_txt, dir):

    replacements = load_replacements(f'{dir}/replacements.txt')

    if 'text-based' in dir:
        words = word_tokenize(raw_txt)
        corrected_words = correct_spelling(words)
        words = [stemmer.stem(w) for w in corrected_words]
        # Argumentation based
    else:
        # Remove patterns from replacements.txt
        for pattern in replacements:
            raw_txt = re.sub(pattern, '', raw_txt)

        raw_txt = replace_money_amounts(raw_txt)

        # Extra stops
        raw_txt = re.sub(r"\.{2,}", ".", raw_txt)

        # Replace sentences: 'hello.World' -> 'hello. World'
        raw_txt = fix_punctuation(raw_txt)
    
        # Weird space, : and -
        raw_txt = re.sub(r' |:|—|-|–', ' ', raw_txt)

        # New lines
        raw_txt = re.sub(r'\n', ' ', raw_txt)

        # With spaces: - and ”
        raw_txt = re.sub(r',”|” |,', ' ', raw_txt)

        # Quotation issues
        raw_txt = re.sub(r"’|'|\"", "'", raw_txt)
        raw_txt = re.sub(r'’’| ”|\$|‘|”', '', raw_txt)

        # Double space
        raw_txt = re.sub(r"\ {2,}", " ", raw_txt)

        # Words with punctuation, and lowercase
        raw_txt = re.sub(r' u.s. ', " united states ", raw_txt.lower())
        raw_txt = re.sub(r' gen. ', " general ", raw_txt)
        raw_txt = re.sub(r' sen. ', " senator ", raw_txt)
        raw_txt = re.sub(r' mr. ', " mister ", raw_txt)
        raw_txt = re.sub(r' gop ', " republican party ", raw_txt)
        raw_txt = re.sub(r' h. w. bush ', " herbert walker bush ", raw_txt)
        raw_txt = re.sub(r' v. ', " versus ", raw_txt)

        raw_txt = re.sub(r' jan. ', " january ", raw_txt)
        raw_txt = re.sub(r' feb. ', " february ", raw_txt)
        raw_txt = re.sub(r' mar. ', " march ", raw_txt)
        raw_txt = re.sub(r' apr. ', " april ", raw_txt)
        raw_txt = re.sub(r' jun. ', " june ", raw_txt)
        raw_txt = re.sub(r' jul. ', " july ", raw_txt)
        raw_txt = re.sub(r' aug. ', " augustus ", raw_txt)
        raw_txt = re.sub(r' sept. ', " september ", raw_txt)
        raw_txt = re.sub(r' oct. ', " october ", raw_txt)
        raw_txt = re.sub(r' nov. ', " november ", raw_txt)
        raw_txt = re.sub(r' dec. ', " december ", raw_txt)

        raw_txt = re.sub(r' n.c. ', " north carolina ", raw_txt)
        raw_txt = re.sub(r' f.b.i ', " federal bureau of investigation ", raw_txt)

        raw_txt = re.sub(r' d.c. ', " district of columbia ", raw_txt)
        raw_txt = re.sub(r' jr. ', " junior ", raw_txt)
        raw_txt = re.sub(r' inc. ', " incorporated ", raw_txt)
        raw_txt = re.sub(r' donald j. trump ', " donald trump ", raw_txt)
        raw_txt = re.sub(r' > ', " over ", raw_txt)
        # raw_txt = re.sub(r'n t. ', " not. ", raw_txt)
        raw_txt = re.sub(r' ll ', " will ", raw_txt)

        raw_txt = re.sub(r', ', ' and ', raw_txt)
        
        # Final left over '
        raw_txt = re.sub(r"' |'", " ", raw_txt)
        raw_txt = re.sub(r"'", "", raw_txt)

        # Contractions
        sentences = []
        for sent in raw_txt.split('. '):
            if len(sent.split()) > 3:
                s = re.sub(r"could ", "could not ", sent.lower())
                s = re.sub(r"i'm ", "i am ", s)

                s = re.sub(r"it s ", "it is ", s)
                s = re.sub(r"mam ", "madame ", s)
                # s = re.sub(r" n t ", " not ", s)

                s = re.sub(r"we re ", "we are ", s)

                s = re.sub(r"there's ", "there is ", s)
                s = re.sub(r"there s ", "there is ", s)

                # Single s
                s = re.sub(r" s ", "s ", s)
                s = re.sub(r"'s ", "s ", s)

                # Extra question marks
                s = re.sub(r"\?{2,}", "?", s)

                # Double spaces
                s = re.sub(r"\ {2,}", " ", s)

                sentences.append(s.strip())

        clean_txt = '. '.join(sentences)

    # Make sure all txts end with a full stop
    if not clean_txt.endswith('.'):
        clean_txt = clean_txt + '.'

    return clean_txt