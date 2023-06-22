# loader.py
#
# Mayla Kersten
#
# This scripts runs loads and cleans the data
# Usage: python loader.py
#

from sklearn.model_selection import train_test_split
import pandas as pd
import os
import cleaner

def load_data(original_dir: str, clean_dir: str):
    """
    Loads the data and calls cleaning function
    Args:
        - original_dir:str,  containing the file in which the data is saved
        - clean_dir: str, containing location of clean data
    """

    for name in os.listdir(f"{original_dir}"):
        if not name == ".DS_Store":
            print(f"{clean_dir}/{name}")
            if os.path.exists(f"{clean_dir}/{name}"):
                print(f"LOAD CLEAN DATASET: {name}")
                print("------------------------------------------------------------------------")

                files = []
                for f in os.listdir(f"{clean_dir}/{name}"):
                    files.append(pd.read_csv(f"{clean_dir}/{name}/{f}", index_col="ID"))

                train, test, validation = files        
            else:
                print(f"CLEAN DATASET: {name}")
                print("------------------------------------------------------------------------")

                if name == "fake_real":
                    df = load_fake(f"{original_dir}/{name}/{name}.csv")
                elif name == "liar":
                    df = load_liar(f"{original_dir}/{name}")
                elif name == "kaggle":
                    df = load_kaggle(f"{original_dir}/{name}")
                else:
                    break
                df.loc[:, 'text'] = df.apply(lambda x: cleaner.clean_text(x["text"], clean_dir), axis=1)
    
                train, validation = train_test_split(df, test_size=0.2)
                validation, test = train_test_split(validation, test_size=0.25)

                print(f"TRAIN - # FAKE: {len(train[train['label'] == 'FAKE'])} - # REAL:{len(train[train['label'] == 'REAL'])}")
                print("------------------------------------------------------------------------")

                if not os.path.exists(f"{clean_dir}/{name}"):
                    os.makedirs(f"{clean_dir}/{name}")
                
                train.to_csv(f"{clean_dir}/{name}/train.csv", columns=["text", "label"], index_label="ID")
                test.to_csv(f"{clean_dir}/{name}/test.csv",  columns=["text", "label"], index_label="ID")
                validation.to_csv(f"{clean_dir}/{name}/validation.csv",  columns=["text", "label"], index_label="ID")

def load_fake(path):
    # Load Fake and Real News dataset by Mcintire
    fake_real = pd.read_csv(path, skip_blank_lines=True)

    # Remove metadata from datasets
    fake_real = fake_real.drop(columns=["idd", "title"])

    # Drop empty rows
    fake_real.dropna(inplace=True)

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

    liar.dropna(inplace=True)

    return liar

def load_kaggle(path):
    """
    Currently capped at 4.000
    """
    df_real = pd.read_csv(f"{path}/True.csv")
    df_real["label"] = "REAL"

    df_fake = pd.read_csv(f"{path}/Fake.csv")
    df_fake["label"] = "FAKE"

    kaggle = pd.concat([df_real, df_fake], ignore_index=True)

    kaggle = kaggle.sample(n=4000, random_state=42, replace=False)
    
    kaggle = kaggle[["text", "label"]]

    kaggle = kaggle.drop_duplicates(subset=["text"])

    kaggle.dropna(inplace=True)
  
    return kaggle

if __name__ == "__main__":
    load_data(original_dir="data", clean_dir="pipeline/text-based/data")
    load_data(original_dir="data", clean_dir="pipeline/argumentation-based/data")