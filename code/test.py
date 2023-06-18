# Run predictions on test set
from collections import defaultdict
import evaluate
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
import torch
import json
from datetime import datetime


all_models = ["elmo"]
#["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "google/electra-base-discriminator", "elmo"]
mode = "text-based"

def predict(row):
    tokenized_text = tokenizer(row['text'], truncation=True,  return_tensors="pt")
    outputs = model(**tokenized_text)
    predicted_class = outputs.logits.argmax().item()

    return predicted_class    

if __name__ == "__main__":
    metric = evaluate.combine(["accuracy", "precision", "recall", "f1"])
    
    json_output = defaultdict(dict)
    for model_name in all_models:
        current_time = datetime.now()

        print(f"{model_name} - START: {current_time.hour}:{current_time.minute}")
        
        # if mode == "argumentation-based":
        #     path = f"models/{mode}/argumentation structure/{model_name}"

        performance = {}
        path = f"pipeline/{mode}/data"
        for dataset in os.listdir(path):
            if os.path.isdir(f"{path}/{dataset}") and dataset != "kaggle_1000":

                path_to_model = f"models/{mode}/{model_name}/{dataset}"

                if model_name == "google/electra-base-discriminator":
                    tokenizer = ElectraTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=300, return_tensors="pt")
                    if torch.cuda.is_available():
                        model = ElectraForSequenceClassification.from_pretrained(path_to_model).to("cuda")
                    else:
                        model = ElectraForSequenceClassification.from_pretrained(path_to_model)

                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=300, return_tensors="pt") 
                    if torch.cuda.is_available():
                        model = AutoModelForSequenceClassification.from_pretrained(path_to_model).to("cuda")
                    else:
                        model = AutoModelForSequenceClassification.from_pretrained(path_to_model)

                df_test = pd.read_csv(f"{path}/{dataset}/test.csv").dropna()[:50]
                print(f"DATASET: {dataset} - LENGTH: {len(df_test)}")
                print("------------------------------------------------------------------------")

                # Convert texts to integers
                df_test["label"] = df_test["label"].map({"FAKE": 0, "REAL": 1})

                # Save the ground truth labels
                gold_labels = df_test["label"].values 

                # Remove labels from test set
                df_test.drop("label", axis=1, inplace=True)

                # Make predictions on unlabeled data
                df_test["prediction"] = df_test.apply(lambda row: predict(row), axis=1)

                # Save output
                predictions = df_test["prediction"].values

                results = metric.compute(predictions=predictions, references=gold_labels)

                performance[dataset] = results

                json_output[model_name][dataset] = results

                table = pd.DataFrame(performance)

                table.to_csv(f"pipeline/{mode}/results/csv/{model_name}_{dataset}.csv")

        table = pd.DataFrame(performance)

        table.to_csv(f"pipeline/{mode}/results/csv/{model_name}.csv")

    with open(f"pipeline/{mode}/results/json/performance.json", 'w') as json_file:
        json.dump(json_output, json_file, indent=4)