#https://github.com/allenai/allennlp-template-python-script/blob/master/my_project/model.py
#http://www.realworldnlpbook.com/blog/training-sentiment-analyzer-using-allennlp.html 
#https://github.com/mhagiwara/realworldnlp/blob/master/examples/sentiment/sst_classifier_elmo.py

import argparse
from allennlp.data import DataLoader, DatasetReader, Vocabulary
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.models import Model
from allennlp.training.trainer import Trainer
from allennlp.training import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.callbacks import TensorBoardCallback
from dataset_reader import ClassificationCsvReader
from itertools import chain
from model import LSTM_Classifier
import os
import pandas as pd
from typing import Tuple
import torch

def build_dataset_reader() -> DatasetReader:
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    reader = ClassificationCsvReader(token_indexers={"tokens": elmo_token_indexer})
    return reader

def build_vocab(train_loader, dev_loader) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(
        chain(train_loader.iter_instances(), dev_loader.iter_instances())
    )

def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")

    elmo_embedder = ElmoTokenEmbedder(
        options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
        weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
        requires_grad=False,  # Set to True if you want to fine-tune ELMo
    )
    embedder = BasicTextFieldEmbedder(
        {"tokens": elmo_embedder}
    ) 

    encoder = LstmSeq2VecEncoder(input_size=1024, hidden_size=23, num_layers=2, bidirectional=True)

    return LSTM_Classifier(vocab, embedder, encoder)

def build_data_loaders(
    reader,
    train_data: str,
    validation_data_path: str= None,) -> Tuple[DataLoader, DataLoader]:
    train_loader = MultiProcessDataLoader(
        reader, train_data, batch_size=32, shuffle=True
    )
    dev_loader = MultiProcessDataLoader(
        reader, validation_data_path, batch_size=32, shuffle=False
    )
    return train_loader, dev_loader




def build_trainer(
    model: Model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    callbacks: list,
    name: str) -> Trainer:
    
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)
    
    model = model.to("cuda")

    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=f"models/{mode}/elmo/{name}",
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=10,
        optimizer=optimizer,
        validation_metric="-loss",
        callbacks=callbacks,
        patience=1,

    )
    return trainer


if __name__ == "__main__":
    print("LOAD DATA")
    print("------------------------------------------------------------------------\n")

    parser = argparse.ArgumentParser(description="Training Elmo")
    parser.add_argument('-m', '--mode', choices=['text-based', 'margot', 'dolly'], help="Select mode: 'text-based' for text-based, 'margot' for argumentation-based Margot, 'dolly' for argumentation-based Dolly")

    args = parser.parse_args()

    mode = args.mode

    if args.mode == "text-based":
        mode = "text-based"
        dir = f"pipeline/{mode}/data"
    elif args.mode == "margot":
        mode = "argumentation-based"
        dir = f"pipeline/{mode}/argumentation structure/margot"
        columns = ["ID", "structure", "label"]
    elif args.mode == "dolly":
        mode = "argumentation-based"
        dir = f"pipeline/{mode}/argumentation structure/dolly" 
        # tool = "dolly"

    print(f"MODE {mode}")
    print("------------------------------------------------------------------------\n")

    for name in os.listdir(dir):
        if os.path.isdir(f"{dir}/{name}"):

            print(f"DATASET: {name}")
            print("------------------------------------------------------------------------\n")

            reader = build_dataset_reader()

            train = pd.read_csv(f"{dir}/{name}/train.csv").dropna()
            validation = pd.read_csv(f"{dir}/{name}/validation.csv").dropna()

            if not args.mode != "text-based":
                train = train.loc[:, columns]
                validation = validation.loc[:, columns]

            train_loader, validation_loader = build_data_loaders(
                reader, train, validation
            )

            vocab = build_vocab(train_loader, validation_loader)

            model = build_model(vocab)

            train_loader.index_with(vocab)
            validation_loader.index_with(vocab)

            tensorboard_callback = TensorBoardCallback(serialization_dir=f"models/{mode}/elmo/{name}/")

            callbacks = [tensorboard_callback]

            trainer = build_trainer(model, train_loader, validation_loader, callbacks, name)

            print("START TRAINING")
            trainer.train()
            torch.save(model, f"models/{mode}/elmo/{name}/pytorch_model.bin")
            print("FINISH TRAINING")

