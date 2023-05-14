# Load data (for both classifiers)
import pandas as pd
import os
import cleaner

def load_data(arg=False):
    data = {}
    for name in os.listdir("data/original"):
        if arg:
            if os.path.exists(f"data/clean/arg/{name}.csv"):
                df = pd.read_csv(f"data/clean/arg/{name}.csv")
            else:
                if name == "fake_real":
                    df = load_fake(f"data/original/{name}/{name}.csv")
                elif name == "liar":
                    df = load_liar(f"data/original/{name}")
                elif name == "kaggle":
                    df = load_kaggle(f"data/original/{name}")
    
                df.loc[:, 'text'] = df.apply(lambda x: cleaner.prep_argumentation_based(x["text"]), axis=1) 
                df.to_csv(f"data/clean/arg/{name}.csv")
                df.to_excel(f"data/clean/arg/{name}.xlsx")
        else:
            if os.path.exists(f"data/clean/text/{name}.csv"):
                df = pd.read_csv(f"data/clean/text/{name}.csv")
            else:
                if name == "fake_real":
                    df = load_fake(f"data/original/{name}/{name}.csv")
                elif name == "liar":
                    df = load_liar(f"data/original/{name}")
                elif name == "kaggle":
                    df = load_kaggle(f"data/original/{name}")
    
                df.loc[:,"text"] = df.apply(lambda x: cleaner.prep_text_based(x["text"]), axis=1) 
                df.to_csv(f"data/clean/text/{name}.csv")
                df.to_excel(f"data/clean/text/{name}.xlsx")  
          
        data[name] =  df

    return data


def load_fake(path):
    # Load Fake and Real News dataset by Mcintire
    fake_real = pd.read_csv(path)

    # Remove metadata from datasets
    fake_real = fake_real.drop(columns=["idd", "title"])

    fake_real.loc[:,"text"] = fake_real.apply(lambda x: cleaner.preprocess(x["text"]), axis=1) 

    return fake_real


def load_liar(path):
    # Load Liar dataset by Wang
    labels = ["id", "label", "statement", "subject", "speaker", "job_title", "state_info", "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"]

    liar_train = pd.read_csv(f"{path}/train.tsv", sep="\t", names=labels)
    liar_valid = pd.read_csv(f"{path}/valid.tsv", sep="\t", names=labels)
    liar_test = pd.read_csv(f"{path}/test.tsv", sep="\t", names=labels)

    liar = pd.concat([liar_train, liar_valid, liar_test]).reset_index(drop=True)

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

    liar.loc[:, 'text'] = liar.apply(lambda x: cleaner.preprocess(x["text"]), axis=1)

    return liar


def load_kaggle(path):
    df_real = pd.read_csv(f"{path}/True.csv")
    df_real["label"] = "REAL"

    df_fake = pd.read_csv(f"{path}/Fake.csv")
    df_fake["label"] = "FAKE"

    kaggle = pd.concat([df_real, df_fake], ignore_index=True)
    kaggle = kaggle[["text", "label"]]

    kaggle = kaggle.drop_duplicates(subset=["text"])

    kaggle.loc[:, 'text'] = kaggle.apply(lambda x: cleaner.preprocess(x["text"]), axis=1)
  
    return kaggle