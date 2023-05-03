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

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Use pretrained model vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device:{device}\n')

id2label = {0: "FAKE", 1: "REAL"}
label2id = {"FAKE": 0, "REAL": 1}

# Model in library with sequence classification head, assigning a class to a statement.
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           num_labels=2, 
                                                           id2label=id2label, 
                                                           label2id=label2id)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    print("LOAD DATA")
    
    fake_real, liar, kaggle = loader.load_data("data")
    print(f"Number of examples: {len(fake_real)}")

    print("------------------------------------")

    print("PREPROCESS DATA")

    fake_real = Dataset.from_pandas(fake_real).train_test_split(test_size=0.3, seed=42).class_encode_column("label")

    tokenized = fake_real.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=300)

    accuracy = evaluate.load("accuracy")

    # # Documentation: https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html

    training_args = TrainingArguments(
        output_dir="results/bert-base-uncased/fake_real",
        per_device_train_batch_size=32,
        evaluation_strategy="epoch",
        learning_rate=4e-5, # Initial learning rate for Adam
        weight_decay=0.01, ##?
        adam_epsilon=1e-8, #Default
        num_train_epochs=10
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator= data_collator,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics
        # Default optimiizer is AdamW
        )
    
    print("START TRAINING")
    trainer.train()

    trainer.save_model()

