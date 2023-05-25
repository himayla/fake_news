# Text-based classifier
# Source: https://huggingface.co/docs/transformers/tasks/sequence_classification
# python train.py arg | python train.py text

from datasets import Dataset
from datetime import datetime
import evaluate
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import sys
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import loader
import torch
#from transformers import AdamW

import argparse



all_models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "google/electra-base-discriminator"]
# mode = sys.argv[2]

def preprocess_function(news):
    return tokenizer(news["text"], truncation=True)

def compute_metrics(eval_pred): 
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metrics = metric.compute(predictions=predictions, references=labels)
    return metrics

if __name__ == "__main__":
    print("LOAD DATA")
    print("------------------------------------------------------------------------")

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('-m', '--mode', choices=['t', 'a'], help="Select mode: 't' for text-based, 'a' for argumentation-based")

    args = parser.parse_args()

    mode = args.mode

    if args.mode == "t":
        mode = "text-based"

    print(mode)

    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    for model_name in all_models:
        current_time = datetime.now()

        print(f"TRAINING: {model_name} - START: {current_time.hour}:{current_time.minute}")
        print("------------------------------------------------------------------------")

        dir = f"pipeline/{mode}/data/"
        for name in os.listdir(dir):
            if os.path.isdir(f"{dir}/{name}"):
                train = pd.read_csv(f"{dir}/{name}/train.csv")
                test = pd.read_csv(f"{dir}/{name}/test.csv")
                # try:
                #     # Load checkpoint
                #     files = os.listdir(f"models/{mode}/{model_name}/{name}")
                #     checkpoint_files = [f for f in files if f.startswith("checkpoint")]
                #     if checkpoint_files:
                #         sorted_checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("-")[1]), reverse=True)
                #         highest_checkpoint = sorted_checkpoint_files[0]
                #         model_path = f"models/text/{model_name}/{name}/" + highest_checkpoint 
                # except FileNotFoundError:
                #     model_path = model_name
                print(f"LOADING: {model_name}")

                if model_name == "google/electra-base-discriminator":
                    tokenizer = ElectraTokenizer.from_pretrained(model_name, padding=True, truncation=True, return_tensors="pt")
                    model = ElectraForSequenceClassification.from_pretrained(f"{model_name}", num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1})
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, return_tensors="pt") 
                    model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}", num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1})
        
                print(f"DATASET: {name} - LENGTH TRAIN: {len(train)}")
                print("------------------------------------------------------------------------")

                train = Dataset.from_pandas(train).class_encode_column("label")
                test = Dataset.from_pandas(test).class_encode_column("label")

                tokenized_train = train.map(preprocess_function, batched=True)
                tokenized_test = train.map(preprocess_function, batched=True)

                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

                # Documentation: https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html
                training_args = TrainingArguments(
                    output_dir=f"models/{mode}/{model_name}/{name}",  # Directory where model checkpoints and logs will be saved
                    per_device_train_batch_size=32,  # Batch size for training
                    evaluation_strategy="steps",  # Evaluate the model after every epoch
                    logging_strategy="epoch",  # Log training data stats for loss after every epoch
                    learning_rate=4e-5,  # Learning rate for the optimizer
                    num_train_epochs=10,  # Number of training epochs
                    logging_dir=f"models/{mode}/{model_name}/{name}/logs",  # Directory where training logs will be saved
                    report_to="tensorboard",
                    save_total_limit=5,  # Limit the total number of saved checkpoints
                    #early_stopping_patience=3,  # Stop training if the evaluation metric does not improve for this many evaluations
                    adam_epsilon=1e-8,  # Epsilon value for Adam optimizer
                    load_best_model_at_end=True,  # Load the best model at the end of training
                    metric_for_best_model="eval_loss",  # Metric to monitor for determining the best model
                    greater_is_better=False,  # Specify if a higher value of the metric is better or not
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator= data_collator,
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_test,
                    compute_metrics=compute_metrics,
                    )
                
                writer = SummaryWriter(log_dir=training_args.logging_dir)

                print("START TRAINING")
                trainer.train()
                trainer.save_model()
                trainer.evaluate()
                writer.close()
