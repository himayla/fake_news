from typing import Dict

import torch
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

@Model.register("LSTM_Classifier")
class LSTM_Classifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        #self.accuracy = CategoricalAccuracy()
       # self.f1 = FBetaMeasure()
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1" : F1Measure(positive_label=1)
        }


    def forward(  # type: ignore
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.metrics["accuracy"](logits, label)
            self.metrics["f1"](logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)

        #self._f1(logits, label)

        return output
        
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        accuracy = self.metrics["accuracy"].get_metric(reset=reset)
        metr = self.metrics["f1"].get_metric(reset=reset)

        metrics = {
            "accuracy": accuracy,
            "precision": metr['precision'], #if precision is not None and not isinstance(precision, str) else 0.0,
            "recall": metr['recall'],# if recall is not None and not isinstance(recall, str) else 0.0,
            "f1": metr['f1'] #if f1 is not None and not isinstance(f1, str) else 0.0,
        }

        return metrics