# test.py
#
# Mayla Kersten
#
# Code to evaluate the trained models on test set
# Usage: python code/train.py -m <MODE>
#
import pandas as pd
import json
from train import parse_arguments, load_params
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ElectraTokenizer,
    ElectraForSequenceClassification,
)
from transformers import PreTrainedModel, Tokenizer
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARGUMENTATION_ELEMENT = "evidence"


def predict(row: dict) -> float:
    if mode == "baseline":
        tokenized_text = tokenizer(row["text"], truncation=True, return_tensors="pt")

    else:
        tokenized_text = tokenizer(
            row[ARGUMENTATION_ELEMENT], truncation=True, return_tensors="pt"
        ) 

    outputs = model(**tokenized_text.to(device))
    predicted_class = outputs.logits.argmax().item()

    return predicted_class


def load_trained_model(path: str, params: dict) -> Tuple[PreTrainedModel, Tokenizer]:
    """
    Loads pre-trained language models
    Args:
        - model_name = str: name of model
        - params = dict: dictionary containing parameters
    """
    if model_name == "google/electra-base-discriminator":
        model = ElectraForSequenceClassification.from_pretrained(path, params["lm"])
        tokenizer = ElectraTokenizer.from_pretrained(path, params["tokenizer"])
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path, params["lm"])
        tokenizer = AutoTokenizer.from_pretrained(path, params["tokenizer"])

    model = model.to(device)

    return model, tokenizer


def load_test_data(path: str) -> Tuple[pd.DataFrame, list]:
    df_test = pd.read_csv(path, index_col="ID").dropna()

    # Convert labels to integer
    df_test["label"] = df_test["label"].map({"FAKE": 0, "REAL": 1})

    # Save the ground truth labels
    gold_labels = df_test["label"].values

    # Remove labels from test set
    df_test.drop("label", axis=1, inplace=True)

    return df_test, gold_labels


def evaluate(predictions, labels):
    metrics = evaluate.combine(["accuracy", "precision", "recall", "f1"])

    results = metrics.compute(predictions=predictions, references=gold_labels)

    return results


if __name__ == "__main__":
    mode = parse_arguments()
    params = load_params()

    results = {}
    for model_name in f"{mode}/models":
        model, tokenizer = load_trained_model(model_name)
        df_test, gold_labels = load_test_data(mode)

        # Make predictions on unlabeled data
        df_test["prediction"] = df_test.apply(lambda row: predict(row), axis=1)

        performance = evaluate(df_test["prediction"].values, gold_labels)

        results[model_name] = performance

    with open(f"{mode}/results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)
