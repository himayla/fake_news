# train.py
#
# Mayla Kersten
#
# Code to train text-based classifier for fake news
# Code is adapted from https://huggingface.co/docs/transformers/tasks/sequence_classification
# Usage: python code/train.py -m <MODE>
#
import argparse
from datasets import Dataset
from datetime import datetime
import evaluate
import json
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ElectraTokenizer,
    ElectraForSequenceClassification,
)
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers import PreTrainedModel, Tokenizer
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARGUMENTATION_ELEMENT = "evidence"

all_models = [
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased",
    "google/electra-base-discriminator",
]

class Classifier(Trainer):
    """Class overwrites default function to compute loss from Trainer"""

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False,
    ) -> tuple[float, dict[str, torch.Tensor]] or float:
        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def parse_arguments() -> str:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        choices=["baseline", "margot", "dolly"],
        help="Select mode: 'baseline',\n'margot' for argumentation-based MARGOT,\n'dolly' for argumentation-based Dolly 2.0.\n",
    )
    args = parser.parse_args()
    mode = args.mode

    return mode


def load_data(path: str) -> Tuple[Dataset, Dataset]:
    data = []

    for i in ["train", "validation"]:
        df = pd.read_csv(f"{path}/{i}.csv", index_col="ID")
        dataset = Dataset.from_pandas(df).class_encode_column("label")
        tokenized = dataset.map(preprocess_function, batched=True)
        data.append(tokenized)

    return train, validation


def load_params() -> Dict[str, Any]:
    """
    Loads parameters for pre-trained language model, tokenizer, and trainer
    """
    with open("params.json") as f:
        params = json.loads(f)

    return params


def load_model(model_name: str, params: dict) -> Tuple[PreTrainedModel, Tokenizer]:
    """
    Loads pre-trained language models
    Args:
        - model_name = str: name of model
        - params = dict: dictionary containing parameters
    """
    if model_name == "google/electra-base-discriminator":
        model = ElectraForSequenceClassification.from_pretrained(params["lm"])
        tokenizer = ElectraTokenizer.from_pretrained(params["tokenizer"])
    else:
        model = AutoModelForSequenceClassification.from_pretrained(params["lm"])
        tokenizer = AutoTokenizer.from_pretrained(params["tokenizer"])

    model = model.to(device)

    return model, tokenizer


def preprocess_function(text: pd.DataFrame) -> dict[str, torch.Tensor]:
    """
    Applies preprocessing, encoding the texts
    """
    if mode == "argumentation":
        return tokenizer(text[ARGUMENTATION_ELEMENT], truncation=True)
    else:
        return tokenizer(text["text"], truncation=True)


def compute_metrics(eval_pred: Tuple[torch.Tensor, int]) -> dict:
    """
    Computes the Accuracy, Precision, Recall and F1, performance of model
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metrics = evaluate.combine(["accuracy", "precision", "recall", "f1"])

    return metrics.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    mode = parse_arguments()
    params = load_params()

    for model_name in all_models:
        current_time = datetime.now()

        print(f"START: {current_time.hour}:{current_time.minute}")

        model, tokenizer = load_model(model_name)

        train, validation = load_data(mode)
        params["model"]["output_dir"] = f"{mode}/{ARGUMENTATION_ELEMENT}/{model_name}"

        trainer = Classifier(
            model=model,
            train_dataset=train,
            eval_dataset=validation,
            args=TrainingArguments(params["model"]),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(1, 0)],
        )

        trainer.save_model()
