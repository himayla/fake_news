# Run predictions on test set
import evaluate
import argparse
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
import json
from datetime import datetime
import torch

ELEMENT = 'evidence'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(row):
    if mode.startswith("argumentation-based"):
        tokenized_text = tokenizer(row[ELEMENT], truncation=True, return_tensors="pt") ### Claim, or evidence, or structure
    else:
        tokenized_text = tokenizer(row["text"], truncation=True, return_tensors="pt")
    outputs = model(**tokenized_text.to(device))
    predicted_class = outputs.logits.argmax().item()

    return predicted_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['text-based', 'margot', 'dolly', 'sample'], help="Select mode: 'text-based' for text-based, 'margot' for argumentation-based Margot, 'dolly' for argumentation-based Dolly")

    args = parser.parse_args()

    if args.mode == "text-based":
        mode = "text-based"
        path_to_data = f"pipeline/{mode}/argumentation structure" 
        path_to_models = f"models/{mode}/{args.mode}/best"
        path_to_results = f"pipeline/{mode}/results"
    else:
        mode = "argumentation-based"
        path_to_data = f"pipeline/{mode}/argumentation structure/{args.mode}" 
        path_to_models = f"models/{mode}/{args.mode}/{ELEMENT}/best"
        path_to_results = f"pipeline/{mode}/results/{args.mode}/{ELEMENT}"

    if not os.path.exists(path_to_models):
        os.makedirs(path_to_models)
  
    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results)
        os.makedirs(f"{path_to_results}/csv")
        os.makedirs(f"{path_to_results}/json")

    print(f"MODE {mode}")
    print("------------------------------------------------------------------------\n")

    metric = evaluate.combine(["accuracy", "precision", "recall", "f1"])
    
    performance = {}
    for model_name in os.listdir(path_to_models):
        # Save predictions (for checking)
        current_time = datetime.now()

        print(f"{model_name} - START: {current_time.hour}:{current_time.minute}")
        
        path_to_model = f"{path_to_models}/{model_name}"

        if model_name == "google":
            tokenizer = ElectraTokenizer.from_pretrained(f"{model_name}/electra-base-discriminator", truncation=True, padding='max_length', max_length=300, return_tensors="pt")
            model = ElectraForSequenceClassification.from_pretrained(f"{path_to_model}/electra-base-discriminator")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=300, return_tensors="pt") 
            model = AutoModelForSequenceClassification.from_pretrained(path_to_model)
        
        model.to(device)

        df_test = pd.read_csv(f"{path_to_data}/test.csv").dropna()

        print(f"LENGTH: {len(df_test)}")
        print("------------------------------------------------------------------------")
                
        # Convert labels to integer
        df_test["label"] = df_test["label"].map({"FAKE": 0, "REAL": 1})

        # Save the ground truth labels
        gold_labels = df_test["label"].values 

        # Remove labels from test set
        df_test.drop("label", axis=1, inplace=True)

        # Make predictions on unlabeled data
        df_test["prediction"] = df_test.apply(lambda row: predict(row), axis=1)

        # Write out
        df_test.to_csv(f"{path_to_results}/{model_name}_predictions.csv")

        # Save output
        predictions = df_test["prediction"].values

        results = metric.compute(predictions=predictions, references=gold_labels)

        performance[model_name] = results

        table = pd.DataFrame(performance)

        table.to_csv(f"{path_to_results}/csv/{model_name}_.csv", index_label=model_name)

    with open(f"{path_to_results}/json/performance_test.json", 'w') as json_file:
        json.dump(performance, json_file, indent=4)