# Text-based classifier
# Source: https://huggingface.co/docs/transformers/tasks/sequence_classification

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from transformers import DataCollatorWithPadding

import evaluate
import numpy as np
import torch
import torch.nn as nn

import loader

SEED = 42

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda")

id2label = {0: "FAKE", 1: "REAL"}
label2id = {"FAKE": 0, "REAL": 1}

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           num_labels=2, 
                                                           id2label=id2label, 
                                                           label2id=label2id)

def preprocess_function(examples):
    return tokenizer(examples["statement"], truncation=True)

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    print("LOAD DATA")
    
    corpora = loader.load_data("data")
    print(corpora)

    print("------------------------------------")



    #print("PREPROCESS DATA")

   # data = Dataset.from_pandas(df_liar).train_test_split(test_size=0.3, seed=42).class_encode_column("label")

    # tokenized = data.map(preprocess_function, batched=True)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # accuracy = evaluate.load("accuracy")

    # training_args = TrainingArguments(
    #     output_dir="bert",
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=4,
    #     num_train_epochs=1,
    #     weight_decay=0.01,
    #     save_strategy="epoch",
    #     load_best_model_at_end=True
    #     )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized["train"],
    #     eval_dataset=tokenized["test"],
    #     tokenizer=tokenizer,
    #     data_collator= data_collator,
    #     compute_metrics=compute_metrics
    #     )
    
    # print("START TRAINING")
    # trainer.train()

