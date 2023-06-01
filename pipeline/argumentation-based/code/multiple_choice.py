from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from transformers import TrainingArguments, Trainer
import numpy as np
import pandas as pd


class DataCollatorForMultipleChoice:
    """
        Dynamically pad the inputs for multiple choice received.
        Flattens all model inputs, applies padding, and unflatten results.

        Optional parameters to change:
        - padding
        - max_length
        - pad_to_multiple_of
    """
    def __call__(self, 
                 features,
                 tokenizer):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = tokenizer.pad(
            flattened_features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
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
    df = pd.read_csv(path_to_data)[:3]

    data = Dataset.from_pandas(df).class_encode_column("label")

    return data

def preprocess_function(examples, tokenizer):
    """ 
        Perform preprocessing of the input.

        Args:
            Dataset: Huggingface Dataset containing features

        Returns:
            DatasetDict with the tokenized examples with corresponding input_ids, attention_mask, and labels.
    """
    # options = ['OptionA', 'OptionB', 'OptionC']

    # first = [[i] * 3 for i in examples["FalseSent"]]

    # second = [
    #     [f"{examples[opt][i]}" for opt in options] for i in range(len(examples['claim']))
    # ]

    combined = [i for i in range(len(examples['claim']))]

    # first = sum(first, [])
    # sec = sum(second, [])
    flattened = sum(combined, [])

    # Truncation makes sure to make sure input is not longer than max
    # tokenized_examples = TOKENIZER(first, sec, truncation=True)
    tokenized_examples = tokenizer(flattened, truncation=True)
    res = {k: v for k, v in tokenized_examples.items()}

    return res

    # return {k: [v[i : i + 3] for i in range(0, len(v), 3)] for k, v in tokenized_examples.items()}

def compute_metrics(eval_predictions):
    """
        Compute metrics on the test set.
    """
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {
        "accuracy": (preds == label_ids).astype(np.float32).mean().item()
        # TODO: precision, recall, f1
        }

if __name__ == "__main__":
    path_to_data = "pipeline/argumentation-based/argumentation structure/kaggle-4000.csv"

    data = load_data(path_to_data)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    tokenized = map(preprocess_function(data, tokenizer), batched=True)

    # training_args = TrainingArguments(
    #     output_dir = f"./models/results/{MODEL.base_model_prefix}",
    #     evaluation_strategy = "epoch",
    #     learning_rate = 5e-5,
    #     per_device_train_batch_size = 16,
    #     per_device_eval_batch_size = 16,
    #     num_train_epochs = 1, # Default = 3
    #     weight_decay = 0.01,
    # )

    # trainer = Trainer(
    #     model = MODEL,
    #     args = training_args,
    #     train_dataset = tokenized["train"],
    #     eval_dataset = tokenized["test"],
    #     tokenizer = TOKENIZER,
    #     data_collator = DataCollatorForMultipleChoice(),
    #     compute_metrics = compute_metrics,
    # )

    # trainer.train()