import evaluate
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification

import loader, write_out

all_models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "google/electra-base-discriminator"]

def predict(row):
    tokenized_text = tokenizer(row['text'], truncation=True,  return_tensors="pt")
    outputs = model(**tokenized_text)
    predicted_class = outputs.logits.argmax().item()

    return predicted_class    

if __name__ == "__main__":
    mode = "text"
    data = loader.load_eval("data/original", "data/clean/test")
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    for model_name in all_models:
        print(f"CURRENTLY EVALUATING: {model_name}")
        print("------------------------------------------------------------------------")

        for name in os.listdir("data/original"):
            try:
                files = os.listdir(f"models/{mode}/{model_name}/{name}") 
            except FileNotFoundError:
                pass

            # Get highest checkpoint
            checkpoint_files = [f for f in files if f.startswith("checkpoint")]
            sorted_checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("-")[1]), reverse=True)
            highest_checkpoint = sorted_checkpoint_files[0]

            res = {}
            print(f"PREDICTIONS FOR {name}")
            print("------------------------------------------------------------------------")

            model_path = f"./models/text/{model_name}/{name}/checkpoint-" + highest_checkpoint

            if model_name == "google/electra-base-discriminator":
                tokenizer = ElectraTokenizer.from_pretrained(model_path, padding=256)
                model = ElectraForSequenceClassification.from_pretrained(model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, padding=256)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)

            df = pd.read_csv(f"data/clean/test/{name}.csv")

            gold_labels = df["label"].values
            df.drop("label", axis=1, inplace=True)

            df["prediction"] = df.apply(lambda row: predict(row), axis=1)
            predictions = df["prediction"].values

            results = metric.compute(predictions=predictions, references=gold_labels)

            res[name] = results

        output = pd.DataFrame.from_dict(res)

        write_out.write_data(output, "results/predictions")