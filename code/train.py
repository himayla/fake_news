# Text-based classifier
# Source: https://huggingface.co/docs/transformers/tasks/sequence_classification

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.utils.tensorboard import SummaryWriter

from datasets import Dataset
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import loader

SEED = 42

all_models = ["distilbert-base-uncased"]#, "google/electra-base-discriminator"]
#all_models = ["bert-base-uncased", "roberta-base","distilbert-base-uncased", "google/electra-base-discriminator"]

def preprocess_function(news):
    return tokenizer(news["text"], truncation=True)

def compute_metrics(eval_pred): 
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metr = metric.compute(predictions=predictions, references=labels)
    #acc.append(metr["accuracy"]) # To create plot later
    return metr

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DEVICE: {device}\n')

    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    id2label = {0: "FAKE", 1: "REAL"}
    label2id = {"FAKE": 0, "REAL": 1}

    print("LOAD DATA")
    data = loader.load_data()

    for mod in all_models:
        # accuracies = {mod: []}
        print(f"CURRENTLY TRAINING: {mod}\n")
        print("------------------------------------------------------------------------")

        model_name = mod
        if model_name == "google/electra-base-discriminator":
            tokenizer = ElectraTokenizer.from_pretrained(f"{model_name}", padding=256)
            model = ElectraForSequenceClassification.from_pretrained(f"{model_name}",                                                                     num_labels=2, 
                                                                    id2label=id2label, 
                                                                    label2id=label2id)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding=256) 
            model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}", 
                                                                    num_labels=2, 
                                                                    id2label=id2label, 
                                                                    label2id=label2id)

        for name, df in data.items():
            acc = {name: []}
   
            print(f"Number of examples: {len(df)}")
            print("------------------------------------")

            df = Dataset.from_pandas(df).train_test_split(test_size=0.3, seed=42).class_encode_column("label")
            df["test"].to_csv(f"data/clean/{name}.csv", index=False)

            tokenized = df.map(preprocess_function, batched=True)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            # Documentation: https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html
            training_args = TrainingArguments(
                output_dir=f"models/text/{model_name}/{name}",
                per_device_train_batch_size=32,
                evaluation_strategy="epoch",
                learning_rate=4e-5, # Initial learning rate for Adam
                weight_decay=0.01, ##?
                adam_epsilon=1e-8, #Default
                num_train_epochs=2, #####
                logging_dir=f"models/text/{model_name}/{name}/logs",  # Specify the directory for TensorBoard logs
                report_to="tensorboard",
                #push_to_hub=True
                )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator= data_collator,
                train_dataset=tokenized["train"],
                eval_dataset=tokenized["test"],
                compute_metrics=compute_metrics,
                # Default optimizer is AdamW
                )
            
            writer = SummaryWriter(log_dir=training_args.logging_dir)

            # acc_per_epoch = dict(zip(np.arange(1, len(accuracies) + 1), accuracies))
            # res = pd.DataFrame.from_dict(acc_per_epoch, orient="index")
            # print(res)
            # res.to_csv(f"models/text/{model_name}/Accuracy per epoch.csv")

            print("START TRAINING")
            trainer.train()

            trainer.save_model()
            #trainer.push_to_hub()

            writer.close()
        



    
