from datasets import Dataset
from datetime import datetime
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from torch.utils.tensorboard import SummaryWriter

import loader, writer

all_models = ["bert-base-uncased", "roberta-base","distilbert-base-uncased", "google/electra-base-discriminator"]

def preprocess_function(news):
    return tokenizer(news["text"], truncation=True)

def compute_metrics(eval_pred): 
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    metr = metric.compute(predictions=predictions, references=labels)
    return metr

if __name__ == "__main__":
    print("LOAD DATA")
    print("------------------------------------------------------------------------")
    #data = loader.load_data(arg=True)

    mode = "arg"

    data = loader.load_annotated()
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    for model_name in all_models:
        current_time = datetime.now()

        print(f"TRAINING: {model_name} - START: {current_time.hour}:{current_time.minute}")
        print("------------------------------------------------------------------------")

        if model_name == "google/electra-base-discriminator":
            tokenizer = ElectraTokenizer.from_pretrained(f"{model_name}", padding=256)
            model = ElectraForSequenceClassification.from_pretrained(f"{model_name}", num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1})
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding=256) 
            model = AutoModelForSequenceClassification.from_pretrained(f"{model_name}", num_labels=2, id2label={0: "FAKE", 1: "REAL"}, label2id={"FAKE": 0, "REAL": 1})
        
        for name, df in data.items():
            print(f"DATASET: {name} - LENGTH: {len(df)}")
            print("------------------------------------")

            if name == "kaggle":
                # ONLY REAL!!!!!!!!!!!!!
                print("ERROR")
                print(len(df["label"] == "REAL"))
            
            df = Dataset.from_pandas(df).train_test_split(test_size=0.3, seed=42).class_encode_column("label")
            writer.write_data(df["test"], f"data/clean/test/annotated/{name}", cols=["claim", "evidence", "label"], tsv=True)

            tokenized = df.map(preprocess_function, batched=True)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            training_args = TrainingArguments(
                output_dir=f"models/arg/{model_name}/{name}",
                per_device_train_batch_size=32,
                evaluation_strategy="epoch",
                learning_rate=4e-5, # Initial learning rate for Adam
                weight_decay=0.01,
                adam_epsilon=1e-8,
                num_train_epochs=10,
                logging_dir=f"models/arg/{model_name}/{name}/logs",  # Specify the directory for TensorBoard logs
                report_to="tensorboard",
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

            print("START TRAINING")
            trainer.train()
            trainer.save_model()

            writer.close()
