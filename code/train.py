# Text-based classifier
# Source: https://huggingface.co/docs/transformers/tasks/sequence_classification
# python train.py arg | python train.py text

from datasets import Dataset
from datetime import datetime
import evaluate
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer, DataCollatorWithPadding
import sys
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
# import loader
import torch
import torch.nn as nn
from pynvml import *

#from transformers import AdamW
print(f"Is CUDA available: {torch.cuda.is_available()}")
# True
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# Tesla T4

import argparse

all_models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "google/electra-base-discriminator"]

def preprocess_function(news):
    return tokenizer(news["text"], truncation=True)

def compute_metrics(eval_pred): 
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metrics = metric.compute(predictions=predictions, references=labels)
    return metrics

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

if __name__ == "__main__":
    print("LOAD DATA")
    print("------------------------------------------------------------------------\n")

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('-m', '--mode', choices=['t', 'a'], help="Select mode: 't' for text-based, 'a' for argumentation-based")

    args = parser.parse_args()

    mode = args.mode

    if args.mode == "t":
        mode = "text-based"

    print(f"MODE {mode}")
    print("------------------------------------------------------------------------\n")

    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    for model_name in all_models:
        current_time = datetime.now()

        print(f"TRAINING: {model_name} - START: {current_time.hour}:{current_time.minute}")
        print("------------------------------------------------------------------------\n")

        dir = f"pipeline/{mode}/data"
        for name in os.listdir(dir):
            if os.path.isdir(f"{dir}/{name}"):
                train = pd.read_csv(f"{dir}/{name}/train.csv").dropna()
                test = pd.read_csv(f"{dir}/{name}/test.csv").dropna()

                print(f"DATASET: {name} - LENGTH TRAIN: {len(train)}")
                print("------------------------------------------------------------------------\n")

                train = Dataset.from_pandas(train).class_encode_column("label")
                test = Dataset.from_pandas(test).class_encode_column("label")
                print(train, test)
                print("------------------------------------------------------------------------\n")

                print(f"LOADING: {model_name}")
                print("------------------------------------------------------------------------\n")

                try:
                    # Load checkpoint
                    files = os.listdir(f"models/{mode}/{model_name}/{name}")
                    checkpoint_files = [f for f in files if f.startswith("checkpoint")]
                    if checkpoint_files:
                        sorted_checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("-")[1]), reverse=True)
                        highest_checkpoint = sorted_checkpoint_files[0]
                        model_path = f"models/text/{model_name}/{name}/" + highest_checkpoint
                    else:
                        model_path = model_name
                except FileNotFoundError:
                    model_path = model_name

                if model_name == "google/electra-base-discriminator":
                    tokenizer = ElectraTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=300, return_tensors="pt")
                    model = ElectraForSequenceClassification.from_pretrained(f"{model_path}", num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1}).to("cuda")
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=300, return_tensors="pt") 
                    model = AutoModelForSequenceClassification.from_pretrained(f"{model_path}", num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1}).to("cuda")
                
                print_gpu_utilization()
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

                tokenized_train = train.map(preprocess_function, batched=True)
                tokenized_test = test.map(preprocess_function, batched=True)

                # Documentation: https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html
                training_args = TrainingArguments(
                    output_dir=f"models/{mode}/{model_name}/{name}",  # Directory where model checkpoints and logs will be saved
                    per_device_train_batch_size=32,  # Batch size for training
                    evaluation_strategy="epoch",  # Evaluate the model after every epoch
                    logging_strategy="epoch",  # Log training data stats for loss after every epoch
                    save_strategy="epoch",
                    learning_rate=4e-5,  # Learning rate for the optimizer
                    optim="adamw_torch",
                    num_train_epochs=10,
                    logging_dir=f"models/{mode}/{model_name}/{name}/logs",  # Directory where training logs will be saved
                    report_to="tensorboard",
                    save_total_limit=5,  # Limit the total number of saved checkpoints
                    adam_epsilon=1e-8,  # Epsilon value for Adam optimizer
                    load_best_model_at_end=True,  # Load the best model at the end of training
                    metric_for_best_model="eval_loss",  # Metric to monitor for determining the best model
                    greater_is_better=False,  # Specify if a higher value of the metric is better or not
                )
    
                early_stop = EarlyStoppingCallback(1, 0) # Zero because delta

                class Classifyer(Trainer):
                    def compute_loss(self, model, inputs, return_outputs=False):
                        labels = inputs.get("labels")
                        # forward pass
                        outputs = model(**inputs)
                        logits = outputs.get("logits")

                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                        return (loss, outputs) if return_outputs else loss

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator= data_collator,
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_test,
                    compute_metrics=compute_metrics,
                    callbacks=[early_stop]
                    )
                
                writer = SummaryWriter(log_dir=training_args.logging_dir)

                print("START TRAINING")
                result = trainer.train()
                print_summary(result)
                trainer.save_model()
                trainer.evaluate()
                writer.close()
                print("------------------------------------------------------------------------\n")
                print()

