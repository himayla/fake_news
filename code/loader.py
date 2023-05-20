# Load data (for both classifiers)
import pandas as pd
import os
import cleaner
from datasets import Dataset

def load_data(arg=False):
    data = {}
    for name in os.listdir("data/original"):
        print("------------------------------------")
        print(f"Load {name}...")
        if arg:
            print(f"Mode: argumentation")
            if os.path.exists(f"data/clean/arg/{name}.csv"):
                print("Loading clean data for argumentation...")
                if name == "kaggle":
                    df = pd.read_csv(f"data/clean/arg/kaggle.csv", nrows=4000)# CAPPED AT 4.000
                else:
                    df = pd.read_csv(f"data/clean/arg/{name}.csv")
            else:
                print("Cleaning the data for argumentation-based classifyer...")

                if name == "fake_real":
                    df = load_fake(f"data/original/{name}/{name}.csv")
                elif name == "liar":
                    df = load_liar(f"data/original/{name}")
                elif name == "kaggle":
                    df = load_kaggle(f"data/original/{name}")
    
                df.loc[:, 'text'] = df.apply(lambda x: cleaner.prep_argumentation_based(x["text"]), axis=1) 
                df.to_csv(f"data/clean/arg/{name}.csv")
                df.to_csv(f"data/clean/arg/{name}.tsv")
                df.to_excel(f"data/clean/arg/{name}.xlsx")
        else:
            if os.path.exists(f"data/clean1/text/{name}.csv"):
                print(f"Loading clean data for text-based classifyer...")
                if name == "kaggle":
                    df = pd.read_csv(f"data/clean/text/kaggle.csv", nrows=4000)# CAPPED AT 4.000
                else:
                    df = pd.read_csv(f"data/clean/text/{name}.csv")
            else:
                print("Cleaning data for text-based classifyer...")

                if name == "fake_real":
                    df = load_fake(f"data/original/{name}/{name}.csv")
                elif name == "liar":
                    df = load_liar(f"data/original/{name}")
                elif name == "kaggle":
                    df = load_kaggle(f"data/original/{name}")[:4000]
                    df = df.rename(columns={"Unnamed: 0": "ID"})

    
                df.loc[:,"text"] = df.apply(lambda x: cleaner.prep_text_based(x["text"]), axis=1) 
                df.to_csv(f"data/clean/text/{name}.tsv", sep="\t", columns=["text", "label"], index=False)
                df.to_csv(f"data/clean/text/{name}.csv", columns=["text", "label"], index=False)
                df.to_excel(f"data/clean/text/{name}.xlsx")  
        data[name] =  df
        print("------------------------------------")

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

def load_eval():
    data = {}
    for name in os.listdir("data/original"):
        if os.path.exists(f"data/test/{name}.csv"):
            data[name] = pd.read_csv(f"data/test/{name}.csv")
        else:
            if name == "fake_real":
                df = load_fake(f"data/original/{name}/{name}.csv")
            elif name == "liar":
                df = load_liar(f"data/original/{name}")
            elif name == "kaggle":
                df = load_kaggle(f"data/original/{name}") 

            df = Dataset.from_pandas(df).train_test_split(test_size=0.3, seed=42).class_encode_column("label")

            data = df["test"]

            data.to_csv(f"data/clean/test/{name}.tsv", sep="\t", columns=["text", "label"], index=False)
            data.to_csv(f"data/clean/test/{name}.csv")
            #data.to_excel(f"data/clean/test/{name}.xlsx")

    return data

def load_tsv():
    data = {}
    for file in os.listdir("data/clean/test"): 
        if file.split('.')[1] == 'tsv':
            data[file.split('.')[0]] = pd.read_csv(f"data/clean/test/{file}", sep="\t")
    return data



