# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     MIAR
   Description :
   Author:         killjl
   date：          2021/11/25
-------------------------------------------------
"""

import numpy
import logging
from typing import Dict, List, Any
from overrides import overrides

import torch
import numpy as np
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.attention import CosineAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.nn import RegularizerApplicator, InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, Auc, F1Measure, Metric, BooleanAccuracy, PearsonCorrelation, \
    Covariance
from torch import nn
from torch.nn import Dropout

import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def pad_sequence2len(tensor, dim, max_len) -> torch.LongTensor:
    shape = tensor.size()
    if shape[dim] == max_len:
        return tensor
    if shape[dim] > max_len:
        new_tensor = tensor[:,:50]
        return new_tensor
    pad_shape = list(shape)
    pad_shape[dim] = max_len - shape[dim]
    pad_tensor = torch.zeros(*pad_shape, device=1, dtype=tensor.dtype)
    new_tensor = torch.cat([tensor, pad_tensor], dim)
    return new_tensor

@Model.register("MIAR")
class MIAR(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 pos_tag_embedding: Embedding = None,
                 dropout: float = 0.5,
                 label_namespace: str = "labels",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)
        self._label_namespace = label_namespace
        self._dropout = Dropout(dropout)
        self._text_field_embedder = text_field_embedder
        self._pos_tag_embedding = pos_tag_embedding or None
        representation_dim = self._text_field_embedder.get_output_dim()
        if pos_tag_embedding is not None:
            representation_dim += self._pos_tag_embedding.get_output_dim()
        self._text_cnn = CnnEncoder(representation_dim, 100, ngram_filter_sizes=(2,3))
        lstm_input_dim = representation_dim
        rnn = nn.LSTM(input_size=lstm_input_dim,
                      hidden_size=350,
                      batch_first=True,
                      bidirectional=True)
        self._encoder = PytorchSeq2SeqWrapper(rnn)
        self._rnn_cnn = CnnEncoder(self._encoder.get_output_dim(), 100, ngram_filter_sizes=(2,3))
        
        self._num_class = self.vocab.get_vocab_size(self._label_namespace)
        self._projector = FeedForward(self._text_cnn.get_output_dim() * 3, 2,
                                      [50, self._num_class],
                                      Activation.by_name("sigmoid")(), dropout)
        self._instances_labels = None
        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f-measure": F1Measure(positive_label=vocab.get_token_index("PROBLEM DISCOVERY", "labels")), 
        }                                                 # INFORMATION GIVING, INFORMATION SEEKING, FEATURE REQUEST,                OTHER
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(self, text: Dict[str, torch.LongTensor], labels: torch.IntTensor = None, pos_tag: torch.LongTensor = None) -> Dict[str, Any]:
        output_dict = dict()
        
        text['tokens'] = pad_sequence2len(text['tokens'], -1, 50)
        text_embedder = self._text_field_embedder(text)
        text_embedder = self._dropout(text_embedder)

        if pos_tag is not None and self._pos_tag_embedding is not None:
            pos_tag = pad_sequence2len(pos_tag, -1, 50)
            pos_tags_embedder = self._pos_tag_embedding(pos_tag)
            text_embedder = torch.cat([text_embedder, pos_tags_embedder], -1)

        text_mask = get_text_field_mask(text).float()
        
        cnn_out = self._text_cnn(text_embedder, text_mask)

        rnn_out = self._encoder(text_embedder, text_mask)
        rnn_out = self._rnn_cnn(rnn_out,text_mask)

        alpha = 0.7  #0  1
        beta = 0.8
        gamma = 0.5
        vec_out1 = alpha * cnn_out + (1-alpha) * rnn_out
        vec_out2 = beta * cnn_out - (1-beta) * rnn_out
        vec_out3 = gamma * rnn_out - (1-gamma) * cnn_out
        vec_out = torch.cat([vec_out1, vec_out2, vec_out3], 1)  

        logits = self._projector(vec_out)
        probs = nn.functional.softmax(logits, dim=-1)
        
        output_dict["logits"] = logits
        output_dict["probs"] = probs
        if labels is not None:
            loss = self._loss(logits, labels)
            output_dict['loss'] = loss
            output_dict['label'] = labels
            for metric_name, metric in self._metrics.items():
                metric(logits, labels)
        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()
        metrics['accuracy'] = self._metrics['accuracy'].get_metric(reset)
        precision, recall, fscore = self._metrics['f-measure'].get_metric(reset)
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['fscore'] = fscore
        return metrics
