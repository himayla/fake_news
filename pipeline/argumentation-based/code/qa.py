from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import evaluate
import torch
import torch.nn as nn
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering,  ElectraTokenizer, ElectraForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForSequenceClassification,  ElectraTokenizer, ElectraForSequenceClassification
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, TrainingArguments, Trainer

import numpy as np
import os
import pandas as pd

from transformers.tokenization_utils_base import  PaddingStrategy
from typing import Optional, Union
import torch

def load_data(path_to_data):
    df = pd.read_csv(path_to_data).dropna()

    df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})
    data = Dataset.from_pandas(df).class_encode_column("label")

    return data

def preprocess_function(examples):
    claims = [c.strip() for c in examples["claim"]]
    evidences = [evidence.strip() for evidence in examples["evidence"]]

    inputs = tokenizer(
        claims,
        evidences,
        max_length=384,
        truncation="longest_first",
        return_offsets_mapping=True,
        padding="max_length",
    )

    return inputs

def compute_metrics(eval_pred): 
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metrics = metric.compute(predictions=predictions, references=labels)
    return metrics

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    
if __name__ == "__main__":

    all_models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "google/electra-base-discriminator"]
    tools = ["margot"]#["dolly", "margot"] #### 

    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    for tool in tools:
        dir = f"pipeline/argumentation-based/argumentation structure/{tool}" 
        for model_name in all_models:
            for name in os.listdir(dir):
                if name != "kaggle_4000": #####
                    if os.path.isdir(f"{dir}/{name}"):

                        current_time = datetime.now()

                        print(f"TOOL: {tool} - TRAINING: {model_name} - START: {current_time.hour}:{current_time.minute}")
                        print("------------------------------------------------------------------------\n")

                        train = pd.read_csv(f"{dir}/{name}/train.csv").dropna()
                        test = pd.read_csv(f"{dir}/{name}/test.csv").dropna()
                        train = Dataset.from_pandas(train).class_encode_column("label")
                        test = Dataset.from_pandas(test).class_encode_column("label")

                        print(f"DATASET: {name} - LENGTH TRAIN: {len(train)}")
                        print("------------------------------------------------------------------------\n")

                        try:
                            files = os.listdir(f"models/argumentation-based/qa/{tool}/{model_name}/{name}")
                            checkpoint_files = [f for f in files if f.startswith("checkpoint")]
                            if checkpoint_files:
                                sorted_checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("-")[1]), reverse=True)
                                highest_checkpoint = sorted_checkpoint_files[0]
                                model_path = f"models/argumentation-based/qa/{tool}/{model_name}/{name}/" + highest_checkpoint
                            else:
                                model_path = model_name
                        except FileNotFoundError:
                            model_path = model_name
                        
                        if model_name == "google/electra-base-discriminator":
                            model = ElectraForSequenceClassification.from_pretrained(f"{model_path}", num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1}).to("cuda")
                            if torch.cuda.is_available():
                                model = model.to("cuda")
                        else:
                            model = AutoModelForSequenceClassification.from_pretrained(f"{model_path}", num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1})
                            if torch.cuda.is_available():
                                model = model.to("cuda")
                                
                        tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=300, return_tensors="pt") 
                        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=300, return_tensors="pt")

                        tokenized_train = train.map(preprocess_function, batched=True)
                        tokenized_test = test.map(preprocess_function, batched=True)

                        print([tokenizer.decode(tokenized_train["input_ids"][42])]) # Sanity check

                        training_args = TrainingArguments(
                            output_dir=f"models/argumentation-based/qa/{tool}/{model_name}/{name}",  # Directory where model checkpoints and logs will be saved
                            per_device_train_batch_size=32,  # Batch size for training
                            evaluation_strategy="epoch",  # Evaluate the model after every epoch
                            logging_strategy="epoch",  # Log training data stats for loss after every epoch
                            save_strategy="epoch",
                            learning_rate=4e-5,  # Learning rate for the optimizer
                            optim="adamw_torch",
                            num_train_epochs=10,
                            logging_dir=f"models/argumentation-based/qa/{tool}/{model_name}/{name}/logs",  # Directory where training logs will be saved
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
                            tokenizer=tokenizer,
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