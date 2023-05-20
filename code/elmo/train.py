#https://github.com/allenai/allennlp-template-python-script/blob/master/my_project/model.py
#http://www.realworldnlpbook.com/blog/training-sentiment-analyzer-using-allennlp.html #https://github.com/mhagiwara/realworldnlp/blob/master/examples/sentiment/sst_classifier_elmo.py

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
from itertools import chain
from typing import Tuple
import sys
import os
from dataset_reader import ClassificationTsvReader
from model import LSTM_Classifier

path = os.path.abspath("/home/mkersten/fake_news/code")
sys.path.append(path)

import loader

mode = sys.argv[1]

def build_dataset_reader() -> DatasetReader:
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    reader = ClassificationTsvReader(token_indexers={"tokens": elmo_token_indexer})
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
    train_data_path: str,
    validation_data_path: str,) -> Tuple[DataLoader, DataLoader]:
    train_loader = MultiProcessDataLoader(
        reader, train_data_path, batch_size=32, shuffle=True
    )
    dev_loader = MultiProcessDataLoader(
        reader, validation_data_path, batch_size=32, shuffle=False
    )
    return train_loader, dev_loader

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    
    # There are a *lot* of other things you could configure with the trainer.  See
    # http://docs.allennlp.org/master/api/training/trainer/#gradientdescenttrainer-objects for more
    # information.

    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=10,
        optimizer=optimizer,
        validation_metric="+accuracy",
    )
    return trainer


def run_training_loop(serialization_dir: str, name: str):
    reader = build_dataset_reader()

    train_data = f"data/clean/{mode}/{name}.tsv"
    test_data = f"data/clean/test/{name}.tsv"

    train_loader, dev_loader = build_data_loaders(
        reader, train_data, test_data
    )

    vocab = build_vocab(train_loader, dev_loader)
    model = build_model(vocab)

    # This is the allennlp-specific functionality in the Dataset object;
    # we need to be able convert strings in the data to integers, and this
    # is how we do it.
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)

    # NOTE: Training using multiple GPUs is hard in this setting.  If you want multi-GPU training,
    # we recommend using our config file template instead, which handles this case better, as well
    # as saving the model in a way that it can be easily loaded later.  If you really want to use
    # your own python script with distributed training, have a look at the code for the allennlp
    # train command (https://github.com/allenai/allennlp/blob/master/allennlp/commands/train.py),
    # which is where we handle distributed training.  Also, let us know on github that you want
    # this; we could refactor things to make this usage much easier, if there's enough interest.

    print("Starting training")
    trainer.train()
    print("Finished training")

if __name__ == "__main__":
    # data = loader.load_data_text(f"data/original", "data/clean") # Uncomment if newer data
    data = loader.load_tsv(f"data/clean/{mode}")

    for name, df in data.items():
        run_training_loop(serialization_dir=f"results/{name}/", name=name)