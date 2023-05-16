import loader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import os
import torch
import pandas as pd
import evaluate
from tqdm import tqdm

model_name = "bert-base-uncased"

path = f"./results/{model_name}/fake_real/checkpoint-1000/"

model = AutoModelForSequenceClassification.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(model_name) 

metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def predict(row):
    tokenized_text = tokenizer(row['text'], truncation=True,  return_tensors="pt")
    outputs = model(**tokenized_text)
    predicted_class = outputs.logits.argmax().item()

    return predicted_class
    

if __name__ == "__main__":
    tqdm.pandas()

    data = loader.load_eval()

    res = {}

    for name in os.listdir("data/original"):
        df = pd.read_csv(f"data/clean/test/{name}.csv")

        print(f"Data: {name}, {len(df)}")
        print("------------------------------------")

        gold_labels = df["label"].values
        df.drop("label", axis=1, inplace=True)

        df["prediction"] = df.progress_apply(lambda row: predict(row), axis=1)
        predictions = df["prediction"].values

        results = metric.compute(predictions=predictions, references=gold_labels)

        res[name] = results

    output = pd.DataFrame.from_dict(res)
    print(output)
    output.to_excel("RESULTS.xlsx")

