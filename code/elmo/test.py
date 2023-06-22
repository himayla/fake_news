#
# Code to make predictions on test set with ELMo
# Tutorial from https://guide.allennlp.org/training-and-prediction#4
from collections import defaultdict
import evaluate
from allennlp.data import DatasetReader, Instance
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.predictors import Predictor
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.predictors import Predictor
import json
import os
import pandas as pd
import torch
import numpy as np

MODE = 'text-based'
metric = evaluate.combine(["accuracy", "precision", "recall", "f1"])


class TestDataReader(DatasetReader):
    def __init__(
        self,
        tokenizer = None,
        token_indexers = None,
        max_tokens: int = 300,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields = {"text": text_field}

        return Instance(fields)
    
    def _read(self, df):
        for i in range(len(df)):
            yield self.text_to_instance(df.loc[i, 'text'])

class FakeNewsClassifierPredictor(Predictor):
    def predict(self, text: str):
        return self.predict_json({"text": text})

    def _json_to_instance(self, json_dict):
        sentence = json_dict["text"]
        return self._dataset_reader.text_to_instance(sentence)

def build_dataset_reader():
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    reader = TestDataReader(token_indexers={"tokens": elmo_token_indexer})
    return reader

def build_data_loader(reader, test_data):
    test_loader = MultiProcessDataLoader(
        reader, test_data, batch_size=32, shuffle=False
    )
    return test_loader

json_output = defaultdict(dict)
performance = {}
path = f"pipeline/{MODE}/data"
for dataset in os.listdir(path):
    if os.path.isdir(f"{path}/{dataset}") and dataset == 'fake_real_1000':
        # Load the fine-tuned model
        path_to_model = f"models/{MODE}/best/elmo/{dataset}/pytorch_model.bin"

        model = torch.load(path_to_model, map_location=torch.device('cpu'))

        # print(model)
        vocab = model.vocab

        # # Put model in evaluation mode
        model.eval()

        # Load test set
        df_test = pd.read_csv(f"{path}/{dataset}/test.csv").dropna()[:25]
        print(f"DATASET: {dataset} - LENGTH: {len(df_test)}")
        print("------------------------------------------------------------------------")
        
        # Preprocess data
        df_test["label"] = df_test["label"].map({"FAKE": 0, "REAL": 1})

        # Save the ground truth labels and drop from dataset
        gold_labels = df_test["label"].values 
        df_test.drop("label", axis=1, inplace=True)

        # Tokenize data
        reader = build_dataset_reader()

        test_loader = build_data_loader(reader, df_test)

        test_loader.index_with(vocab)

        predictor = FakeNewsClassifierPredictor(model, reader)

        counter = 0
        all_predictions = []
        for instance in test_loader.iter_instances():
            print(counter)
            if counter in np.arange(2, 50, 2):
                results = metric.compute(predictions=all_predictions, references=gold_labels[:counter])

                table = pd.DataFrame(results, index=[0]).T
                table.to_csv(f"pipeline/{MODE}/results/csv/temp_elmo_{dataset}.csv", index_label='ELMo')

            output = predictor.predict_instance(instance)
            labels = [(vocab.get_token_from_index(label_id, "labels"), prob) for label_id, prob in enumerate(output["probs"])]
            max_prediction = max(labels, key=lambda x: x[1])
            predicted_label = max_prediction[0]
            if predicted_label == 'REAL':
                all_predictions.append(1)
            else:
                all_predictions.append(0)
            counter += 1
        results = metric.compute(predictions=all_predictions, references=gold_labels)


        performance[dataset] = results

        json_output['elmo'][dataset] = results

        table = pd.DataFrame(performance)

        table.to_csv(f"pipeline/{MODE}/results/csv/elmo_{dataset}.csv")

        table = pd.DataFrame(performance)

        table.to_csv(f"pipeline/{MODE}/results/csv/elmo.csv", index_label='ELMo')

with open(f"pipeline/{MODE}/results/json/performance_elmo.json", 'w') as json_file:
    json.dump(json_output, json_file, indent=4)