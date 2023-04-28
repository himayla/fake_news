# Imports
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download("punkt")
from textblob import TextBlob
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Load NLTK stopwords and Snowball stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer(language="english")

def n_label(df, label):
    """ Returns number of examples with a specific label """
    return len(df[df["label"] == label])

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
    stemmed = [stemmer.stem(w) for w in corrected.split()]

    return ' '.join(stemmed)

if __name__ == "__main__":
    print("Starting program")
    # tqdm.pandas()
    start_time = time.time()
    # Load Fake and Real News dataset by Mcintire
    fake_real = pd.read_csv("data/mcintire/fake_and_real_news_dataset.csv")

    # # Load Liar dataset by Yang
    labels = ["id", "label", "statement", "subject", "speaker", "job_title", "state_info", "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"]

    liar_train = pd.read_csv("data/liar/train.tsv", sep="\t", names=labels)
    liar_valid = pd.read_csv("data/liar/valid.tsv", sep="\t", names=labels)
    liar_test = pd.read_csv("data/liar/test.tsv", sep="\t", names=labels)

    liar = pd.concat([liar_train, liar_valid, liar_test])

    # tqdm.pandas(desc="Test")

    # Load Fake and Real news from Kaggle
    df_real = pd.read_csv("data/kaggle/True.csv")
    df_real["label"] = "REAL"

    df_fake = pd.read_csv("data/kaggle/Fake.csv")
    df_fake["label"] = "FAKE"

    kaggle = pd.concat([df_real, df_fake], ignore_index=True)

    # Convert labels
    liar["label"] = liar["label"].map({
        "true": "REAL",
        "half-true": "REAL",
        "mostly-true": "REAL",
        "barely-true": "FAKE",
        "pants-fire": "FAKE",
        "false": "FAKE"
    })

    # Remove metadata from datasets
    fake_real = fake_real.drop(columns=["idd", "title"])
    liar = liar[["label", "statement"]]
    kaggle = kaggle[["text", "label"]]

    # Rename column to match the other datasets
    liar = liar.rename(columns={"statement": "text"})

    print("Preprocessing Fake and Real News")
    # fake_real = fake_real.loc.copy()
    fake_real.loc[:,"clean_text"] = fake_real.apply(lambda x: preprocess(x["text"]), axis=1)

    print("Preprocessing Liar")
    liar.loc[:,"clean_text"] = liar.apply(lambda x: preprocess(x["text"]), axis=1)

    print("Preprocessing Liar")
    kaggle.loc[:,"clean_text"] = kaggle.apply(lambda x: preprocess(x["text"]), axis=1)

    # Write clean data out
    fake_real.to_csv("data/clean/fake_real.csv")
    liar.to_csv("data/clean/liar.csv")
    kaggle.to_csv("data/clean/kaggle.csv")
    print("--- %s seconds ---" % (time.time() - start_time))

