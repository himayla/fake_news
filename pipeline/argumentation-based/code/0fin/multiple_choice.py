from datasets import Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import evaluate
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMultipleChoice,  ElectraTokenizer, ElectraForMultipleChoice
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
import numpy as np
import os
import pandas as pd

from dataclasses import dataclass
from transformers.tokenization_utils_base import  PaddingStrategy
from typing import Optional, Union
import torch


@dataclass
class DataCollatorForMultipleChoice:
    """
        Data collator that will dynamically pad the inputs for multiple choice received.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = 'max_length'
    max_length: Optional[int] = 312
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        self.tokenizer = tokenizer
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v for k, v in feature.items()}] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = tokenizer.truncate_sequences(
                flattened_features,
                num_tokens_to_remove=len(features) - self.max_length,
                truncation_strategy='longest_first',
        )

        batch = tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def load_data(path_to_data):
    """
        Load and merges sentences and labels, split data, encode labels and convert to Huggingface DatasetDict.

        Args:
            path_to_data = name of file containing the claims, premises, and the correct label.
            write_out = optional argument, to write out the test data for analytics.

        Returns: 
            DatasetDict containing all train and test data. 
    """
    df = pd.read_csv(path_to_data).dropna()[:5]

    df["label"] = df["label"].map({"FAKE": 0, "REAL": 1})
    data = Dataset.from_pandas(df).class_encode_column("label")

    return data

def preprocess_function(examples):
    """ 
        Perform preprocessing of the input per batch.

        Args:
            Dataset: Huggingface Dataset containing features

        Returns:
            DatasetDict with the tokenized examples with corresponding input_ids, attention_mask, and labels.
    """
    # Originally from: https://huggingface.co/docs/transformers/tasks/multiple_choice

    claims = [[i] for i in examples["claim"]]
    evidences = [[i] for i in examples["evidence"]]

    claims = sum(claims, [])
    evidences = sum(evidences, [])

    # Truncation makes sure to make sure input is not longer than max
    tokenized_examples = tokenizer(claims, evidences, truncation=True)

    #r = {k: [v[i : i + 4] for i in range(0, len(v), 4)] }

    return tokenized_examples
    
def compute_metrics(eval_pred): 
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metrics = metric.compute(predictions=predictions, references=labels)
    return metrics

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    
if __name__ == "__main__":
    #rint(f"Is CUDA available: {torch.cuda.is_available()}")
    #True
    #print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    #Tesla T4

    all_models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]#, "google/electra-base-discriminator"]
    tools = ["dolly", "margot"]
    name = "kaggle"

    path_to_train = f"pipeline/argumentation-based/argumentation structure/margot/{name}/train.csv"
    path_to_test = f"pipeline/argumentation-based/argumentation structure/margot/{name}/test.csv"
    
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    for tool in tools:
        for model_name in all_models:
            current_time = datetime.now()

            print(f"TOOL: {tool} - TRAINING: {model_name} - START: {current_time.hour}:{current_time.minute}")
            print("------------------------------------------------------------------------\n")

            train = load_data(path_to_train)
            test = load_data(path_to_test)

            print(f"DATASET: {name} - LENGTH TRAIN: {len(train)}")
            print("------------------------------------------------------------------------\n")

            try:
                files = os.listdir(f"models/argumentation-based/multiple_choice/{tool}/{model_name}/{name}")
                checkpoint_files = [f for f in files if f.startswith("checkpoint")]
                if checkpoint_files:
                    sorted_checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split("-")[1]), reverse=True)
                    highest_checkpoint = sorted_checkpoint_files[0]
                    model_path = f"models/argumentation-based/multiple_choice/{tool}/{model_name}/{name}/" + highest_checkpoint
                else:
                    model_path = model_name
            except FileNotFoundError:
                model_path = model_name

            if model_name == "google/electra-base-discriminator":
                tokenizer = ElectraTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=312, return_tensors="pt")
                model = ElectraForMultipleChoice.from_pretrained(f"{model_path}")#.to("cuda")#, num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1})
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding='max_length', max_length=312, return_tensors="pt") 
                model = AutoModelForMultipleChoice.from_pretrained(f"{model_path}")#.to("cuda")#, num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1})#

            # data_collator = DataCollatorForMultipleChoice(tokenizer)

            tokenized_train = train.map(preprocess_function, batched=True)
            tokenized_test = test.map(preprocess_function, batched=True)

            # SANITY CHECK
            # x = [tokenizer.decode(tokenized_train["input_ids"][0])]# for i in range(4)]
            # print(x)
            # print([tokenizer.decode(tokenized_train["input_ids"][1][i]) for i in range(2)])
            # accepted_keys = ["input_ids", "attention_mask", "label"]
            # features = [{k: v for k, v in tokenized_train[i].items() if k in accepted_keys} for i in range(3)]
            # batch = DataCollatorForMultipleChoice()(features, tokenizer=tokenizer)
            # y = [tokenizer.decode(batch["input_ids"][2][i].tolist()) for i in range(4)]
            # print(y)
            # x = [tokenizer.decode(batch["input_ids"][2].tolist())]
            # print(x)

            training_args = TrainingArguments(
                output_dir=f"models/argumentation-based/multiple_choice/{tool}/{model_name}/{name}",  # Directory where model checkpoints and logs will be saved
                per_device_train_batch_size=32,  # Batch size for training
                evaluation_strategy="epoch",  # Evaluate the model after every epoch
                logging_strategy="epoch",  # Log training data stats for loss after every epoch
                save_strategy="epoch",
                learning_rate=4e-5,  # Learning rate for the optimizer
                optim="adamw_torch",
                num_train_epochs=2,
                logging_dir=f"models/argumentation-based/multiple_choice/{tool}/{model_name}/{name}/logs",  # Directory where training logs will be saved
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
                data_collator= DataCollatorForMultipleChoice(tokenizer),
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