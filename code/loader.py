# loader.py
#
# Mayla Kersten
#
# This scripts loads and cleans the data
# Usage: python loader.py
#
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional

import cleaner

SAMPLE_SIZE = 8500


def load_data(
    original_data: str, clean_data, sample: Optional[bool] = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads clean data or calls cleaning function
    Args:
        - original_data_dir: str, contains the original data directory
        - clean_data_dir: str, contains destination directory for clean data
        - sample: Optional, if wish to use sample of the data
    """

    for dataset in os.listdir(original_data):
        # In case clean directory already exists extract this data
        if os.path.exists(f"{clean_data}/{dataset}"):
            print(f"LOAD CLEAN DATASET: {dataset}")
            print(
                "------------------------------------------------------------------------"
            )

            files = []
            for f in os.listdir(f"{clean_data}/{dataset}"):
                files.append(pd.read_csv(f"{clean_data}/{dataset}/{f}", index_col="ID"))

            train, test, validation = files
        else:
            print(f"CLEAN DATASET: {dataset}")
            print(
                "------------------------------------------------------------------------"
            )

            if dataset == "fake_real":
                df = load_fake(f"{original_data}/{dataset}/{dataset}.csv", sample=True)
            elif dataset == "liar":
                df = load_liar(f"{original_data}/{dataset}", sample=True)
            elif dataset == "kaggle":
                df = load_kaggle(f"{original_data}/{dataset}", sample=True)
            else:
                break

            df.loc[:, "text"] = df.apply(
                lambda x: cleaner.clean_text(x["text"], clean_data), axis=1
            )

            train, validation = train_test_split(df, test_size=0.2)
            validation, test = train_test_split(validation, test_size=0.25)

            os.makedirs(f"{clean_data}/{dataset}")

            train.to_csv(
                f"{clean_data}/{dataset}/train.csv",
                columns=["text", "label"],
                index_label="ID",
            )
            validation.to_csv(
                f"{clean_data}/{dataset}/validation.csv",
                columns=["text", "label"],
                index_label="ID",
            )
            test.to_csv(
                f"{clean_data}/{dataset}/test.csv",
                columns=["text", "label"],
                index_label="ID",
            )

    return train, validation, test


def load_fake(path: str, sample: bool) -> pd.DataFrame:
    """
    Load Fake and Real News dataset by Mcintire
    Args:
        - path: str, path to the dataset
    """
    fake_real = pd.read_csv(path, skip_blank_lines=True)

    if sample == True:
        fake_real = fake_real.sample(n=SAMPLE_SIZE / 3, replace=False)

    # Remove metadata from datasets
    fake_real = fake_real.drop(columns=["idd", "title"])

    # Drop empty rows
    fake_real.dropna(inplace=True)

    return fake_real


def load_liar(path: str, sample: bool):
    """
    Load Liar dataset by Wang
    Args:
        - path: str, path to the dataset
        - sample: bool, whether or not to subsample the dataset
    """
    # Use column names to read the TSV files
    columns = [
        "id",
        "label",
        "statement",
        "subject",
        "speaker",
        "job_title",
        "state_info",
        "party_affiliation",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context",
    ]

    liar_train = pd.read_csv(f"{path}/train.tsv", sep="\t", names=columns)
    liar_valid = pd.read_csv(f"{path}/valid.tsv", sep="\t", names=columns)
    liar_test = pd.read_csv(f"{path}/test.tsv", sep="\t", names=columns)

    liar = pd.concat([liar_train, liar_valid, liar_test]).reset_index(drop=True)

    if sample == True:
        liar = liar.sample(n=SAMPLE_SIZE / 3, replace=False)

    # Convert labels
    liar["label"] = liar["label"].map(
        {
            "true": "REAL",
            "half-true": "REAL",
            "mostly-true": "REAL",
            "barely-true": "FAKE",
            "pants-fire": "FAKE",
            "false": "FAKE",
        }
    )

    # Drop additional metadata
    liar = liar[["statement", "label"]]

    # Rename statement to text for consistency
    liar = liar.rename(columns={"statement": "text"})

    # Drop empty rows
    liar.dropna(inplace=True)

    return liar


def load_kaggle(path: str, sample: bool):
    """
    Load Liar dataset by Bisaillion
    Args:
        - path: str, path to the dataset
        - sample: bool, whether or not to subsample the dataset
    """
    df_real = pd.read_csv(f"{path}/True.csv")
    df_real["label"] = "REAL"

    df_fake = pd.read_csv(f"{path}/Fake.csv")
    df_fake["label"] = "FAKE"

    kaggle = pd.concat([df_real, df_fake], ignore_index=True)

    if sample == True:
        kaggle = kaggle.sample(n=SAMPLE_SIZE / 3, replace=False)

    kaggle = kaggle[["text", "label"]]

    kaggle = kaggle.drop_duplicates(subset=["text"])

    kaggle.dropna(inplace=True)

    return kaggle


if __name__ == "__main__":
    load_data(original_dir="data", clean_dir="pipeline/text-based/data")
    # load_data(original_dir="data", clean_dir="pipeline/argumentation-based/data")
