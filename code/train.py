# Text-based classifier
# Source: https://huggingface.co/docs/transformers/tasks/sequence_classification
# Usage: python train.py arg | python train.py text

import argparse
from datasets import Dataset
from datetime import datetime
import evaluate
import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

all_models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "google/electra-base-discriminator"]

def preprocess_function(news):
    if mode.startswith("argumentation-based"):
        return tokenizer(news[specs], truncation=True) ### Claim, or evidence, or structure
    else:
        return tokenizer(news["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    all_predictions.append(predictions)

    metrics = metric.compute(predictions=predictions, references=labels)
    return metrics


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['text-based', 'margot', 'dolly'], help="Select mode: 'text-based' for text-based, 'margot' for argumentation-based Margot, 'dolly' for argumentation-based Dolly")

    args = parser.parse_args()

    mode = args.mode

    if args.mode == "text-based":
        path = f"pipeline/{mode}/data"
    elif args.mode == "margot":
        specs = "structure"
        mode = f"argumentation-based"
        path = f"pipeline/{mode}/argumentation structure/margot"
    elif args.mode == "dolly":
        mode = "argumentation-based"
        path = f"pipeline/{mode}/argumentation structure/dolly" 


    print(f"MODE {mode}")
    print("------------------------------------------------------------------------\n")

    metric = evaluate.combine(["accuracy", "precision", "recall", "f1"])

    for model_name in all_models:
        current_time = datetime.now()

        print(f"{model_name} - START: {current_time.hour}:{current_time.minute}")
        print("------------------------------------------------------------------------\n")

        for name in os.listdir(path):
            if os.path.isdir(f"{path}/{name}"):
                
                df_train = pd.read_csv(f"{path}/{name}/train.csv").dropna()
                df_val = pd.read_csv(f"{path}/{name}/validation.csv").dropna()

                print(f"DATASET: {name}")
                print("------------------------------------------------------------------------\n")

                train = Dataset.from_pandas(df_train).class_encode_column("label")
                validation = Dataset.from_pandas(df_val).class_encode_column("label")

                print(f"LOADING: {model_name}")
                print("------------------------------------------------------------------------\n")

                try:
                    files = os.listdir(f"models/{mode}/{model_name}/{name}")
                    checkpoint_files = [f for f in files if f.startswith("checkpoint")]
                    if checkpoint_files:
                        sorted_checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("-")[1]), reverse=True)
                        highest_checkpoint = sorted_checkpoint_files[0]
                        model_path = f"models/{mode}/{model_name}/{name}/" + highest_checkpoint
                    else:
                        model_path = model_name
                except FileNotFoundError:
                    model_path = model_name

                if model_name == "google/electra-base-discriminator":
                    tokenizer = ElectraTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=300, return_tensors="pt")
                    model = ElectraForSequenceClassification.from_pretrained(f"{model_path}", num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1})
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=300, return_tensors="pt") 
                    model = AutoModelForSequenceClassification.from_pretrained(f"{model_path}", num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1})
                if torch.cuda.is_available():
                    model = model.to("cuda")
      
                # Tokenize data
                tokenized_train = train.map(preprocess_function, batched=True)
                tokenized_val = validation.map(preprocess_function, batched=True)

                # Save predictions
                all_predictions = []

                output_path = f"models/{mode}/{model_name}/{name}"

                if args.mode == 'margot':
                    output_path = f"models/{mode}/{specs}/{model_name}/{name}"


                # Documentation: https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html
                training_args = TrainingArguments(
                    output_dir=output_path,  # Directory where model checkpoints and logs will be saved
                    per_device_train_batch_size=32,  # Batch size for training
                    evaluation_strategy="epoch",  # Evaluate the model after every epoch
                    logging_strategy="epoch",  # Log training data stats for loss after every epoch
                    learning_rate=4e-5,  # Learning rate for the optimizer
                    save_strategy="epoch",
                    optim="adamw_torch",
                    num_train_epochs=10,
                    logging_dir=f"{output_path}/logs",  # Directory where training logs will be saved
                    report_to="tensorboard",
                    adam_epsilon=1e-8,  # Epsilon value for Adam optimizer
                    load_best_model_at_end=True,  # Load the best model at the end of training
                    metric_for_best_model="eval_loss",  # Metric to monitor for determining the best model
                    save_total_limit = 1
                )
    
                early_stop = EarlyStoppingCallback(1, 0) # Zero because delta

                class Classifier(Trainer):
                    def compute_loss(self, model, inputs, return_outputs=False):
                        labels = inputs.get("labels")
                        # Forward pass
                        outputs = model(**inputs)
                        logits = outputs.get("logits")

                        loss_fct = nn.CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                        return (loss, outputs) if return_outputs else loss

                trainer = Classifier(
                    model=model,
                    args=training_args,
                    data_collator= DataCollatorWithPadding(tokenizer=tokenizer),
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_val,
                    compute_metrics=compute_metrics,
                    callbacks=[early_stop]
                    )
                
                writer = SummaryWriter(log_dir=training_args.logging_dir)

                print("START TRAINING")
                print("------------------------------------------------------------------------\n")

                result = trainer.train()
                print_summary(result)
                trainer.save_model()
                trainer.evaluate()
                writer.close()


                with open(f"{output_path}/predictions.txt", "w") as f:
                    for idx, pred in enumerate(all_predictions):
                        f.write(f"epoch {idx}: {pred}\n")
                print("------------------------------------------------------------------------\n")
                print()

