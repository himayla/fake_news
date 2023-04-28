import argparse
import pandas as pd
import string
# import nltk
from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
# nltk.download("punkt")
from textblob import TextBlob
import subprocess

tools = ["MARGOT", "ARGMINING-17"]
path = "AM/MARGOT/run_margot.sh" # Default

# Load NLTK stopwords and Snowball stemmer
stop_words = set(stopwords.words('english'))
#stemmer = SnowballStemmer(language="english")

counter = 0 

def parse_commands():
    parser = argparse.ArgumentParser(prog='AM')

    parser.add_argument("--tool", help="Tool to use")

    try:
        args = parser.parse_args()
        if args.tool == tools[0]:
            pass
        else:
            path = ""
    except:
        pass

def preprocess(raw_text):
    """ Perform preprocessing"""
    # Remove urls and IPs
    txt = raw_text.replace(r"http://\S+|https://\S++", "").replace(r"\.[0]*", "")

    word_tokens = word_tokenize(txt)

    # Remove stop words 
    sent = [w for w in word_tokens if not w.lower() in stop_words]

    # Replace different colons to regular ones
    sent = [w.replace("”", "\"").replace("“", "\"").replace("’", "\"").replace("...", ".") for w in sent]

    # Remove punctuation and split every text by white space
    sent = ' '.join([w for w in sent if w not in string.punctuation])

    # Correct spelling of words
    doc = TextBlob(sent)
    corrected = doc.correct()

    # Remove suffices by stemming
    #stemmed = [stemmer.stem(w) for w in corrected.split()]

    return ''.join(corrected)

def MARGOT(clean_txt):
    with open(f"AM/MARGOT/tmp/temp_{counter}.txt", 'w') as f:
        #txt = clean_txt.to_string(header=False, index=False)
        f.writelines(clean_txt)

    subprocess.call([path, f"tmp/temp_{counter}.txt", "../../data/MARGOT"])

if __name__ == "__main__":
    parse_commands()

    fake_real = pd.read_csv("data/clean/fake_real.csv")[:1]

    example = fake_real["text"][0]

    clean_example = preprocess(example)

    fake_real["clean"] = fake_real.apply(lambda x: preprocess(x["text"]), axis=1)


    #print(example)

    rc = fake_real.apply(lambda x: MARGOT(x["clean"]), axis=1)

    #print(rc)

    # pd.write_json(rc)