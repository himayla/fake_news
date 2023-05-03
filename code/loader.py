import pandas as pd
import os.path
import cleaner

def load_data(dir):
    if (len(os.listdir(f"{dir}/clean")) == 0):
        fake_real = load_fake(f"{dir}/mcintire/fake_real.csv")
        fake_real.to_csv(f"{dir}/clean/fake_real.csv")

        liar = load_liar(f"{dir}/liar")
        liar.to_csv(f"{dir}/clean/liar.csv")

        kaggle = load_kaggle(f"{dir}/kaggle")
        kaggle.to_csv(f"{dir}/clean/kaggle.csv")
       
    else:
        fake_real = pd.read_csv(f"{dir}/clean/fake_real.csv")
        liar = pd.read_csv(f"{dir}/clean/liar.csv")
        kaggle = pd.read_csv(f"{dir}/clean/kaggle.csv")
    

    return fake_real, liar, kaggle

def load_fake(path):
    # Load Fake and Real News dataset by Mcintire
    fake_real = pd.read_csv(path)
    # Remove metadata from datasets
    fake_real = fake_real.drop(columns=["idd", "title"])

    fake_real.apply(lambda x: cleaner.preprocess(x["text"]), axis=1) 

    return fake_real

def load_liar(path):
    # Load Liar dataset by Wang
    labels = ["id", "label", "statement", "subject", "speaker", "job_title", "state_info", "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"]

    liar_train = pd.read_csv(f"{path}/train.tsv", sep="\t", names=labels)
    liar_valid = pd.read_csv(f"{path}/valid.tsv", sep="\t", names=labels)
    liar_test = pd.read_csv(f"{path}/test.tsv", sep="\t", names=labels)

    liar = pd.concat([liar_train, liar_valid, liar_test])

    # Convert labels
    liar["label"] = liar["label"].map({
        "true": "REAL",
        "half-true": "REAL",
        "mostly-true": "REAL",
        "barely-true": "FAKE",
        "pants-fire": "FAKE",
        "false": "FAKE"
    })

    liar = liar[["label", "statement"]]

    liar = liar.rename(columns={"statement": "text"})

    liar.apply(lambda x: cleaner.preprocess(x["text"]), axis=1)

    return liar

def load_kaggle(path):
    df_real = pd.read_csv(f"{path}/True.csv")
    df_real["label"] = "REAL"

    df_fake = pd.read_csv(f"{path}/Fake.csv")
    df_fake["label"] = "FAKE"

    kaggle = pd.concat([df_real, df_fake], ignore_index=True)
    kaggle = kaggle[["text", "label"]]

    kaggle = kaggle.drop_duplicates(subset=["text"])

    kaggle.apply(lambda x: cleaner.preprocess(x["text"]), axis=1)
  
    return kaggle