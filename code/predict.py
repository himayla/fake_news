from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import torch
import pandas as pd
import evaluate
from tqdm import tqdm
import loader

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name) 

checkpoints = {"fake_real": '1000/', "kaggle": '1500/', "liar": '2500/'}

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
        checkpoint = checkpoints[name]
        path = f"./models/text/{model_name}/{name}/checkpoint-" + checkpoint

        model = AutoModelForSequenceClassification.from_pretrained(path)
        
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
    output.to_csv("RESULTS.xlsx")