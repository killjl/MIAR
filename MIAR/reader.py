# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     review_reader
   Description :
   Author :        killjl
   date：          2021/11/25
-------------------------------------------------
"""
import json
import random
import re
from collections import defaultdict
from itertools import permutations
from typing import Dict, List
import logging

from allennlp.data import Field
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers.word_stemmer import PorterStemmer
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, field, text_field, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, token
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("review_reader")
class ReviewReader(DatasetReader):
    """
    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the sentence into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """

    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 segment_sentences: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True),
                                                     word_stemmer=PorterStemmer())
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if segment_sentences:
            self._segment_sentences = SpacySentenceSplitter()
        self._class_cnt = defaultdict(int)

    def read_dataset(self, file_path):
        all = []
        features = []
        bugs = []
        ratings = []
        UEs = []
        others = []
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            i = 0
            for line in data_file:
                if i % 2 == 0:
                    cmts = line.strip('\n')
                    all.append(cmts.lower())
                    i += 1
                else:
                    label = line.strip('\n')
                    if label == "INFORMATION GIVING":
                        features.append(all[-1])
                    elif label == "INFORMATION SEEKING":
                        bugs.append(all[-1])
                    elif label == "PROBLEM DISCOVERY":
                        ratings.append(all[-1])
                    elif label == "FEATURE REQUEST":
                        UEs.append(all[-1])
                    elif label == "OTHER":
                        others.append(all[-1])
                    else:
                        logger.info("error. %s, %s",label, all[-1])
                    i += 1
        return features, bugs, ratings, UEs, others


    @overrides
    def _read(self, file_path):
        features, bugs, ratings, UEs, others = self.read_dataset(file_path)
        all_data = features + bugs + ratings + UEs + others
        logger.info(f"ig sample num is {len(features)}")
        logger.info(f"is sample num is {len(bugs)}")
        logger.info(f"pd sample num is {len(ratings)}")
        logger.info(f"fr sample num is {len(UEs)}")
        logger.info(f"other sample num is {len(others)}")
        if "test" in file_path:
            logger.info("Begin predict------")
        elif "val" in file_path:
            logger.info("Begin validation-------")
        else:
            logger.info("Begin training-------")
        for sample in features:
            yield self.text_to_instance(sample, 'INFORMATION GIVING')
        for sample in bugs:
            yield self.text_to_instance(sample, 'INFORMATION SEEKING') #Other
        for sample in ratings:
            yield self.text_to_instance(sample, 'PROBLEM DISCOVERY') #Other
        for sample in UEs:
            yield self.text_to_instance(sample, 'FEATURE REQUEST') #Other
        for sample in others:
            yield self.text_to_instance(sample, 'OTHER') #Other
        logger.info(f"Predict sample num is {len(all_data)}")


    @overrides
    def text_to_instance(self, text:str, label:str = None) -> Instance:  # type: ignore
        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)
        fields = {'text': TextField(tokenized_text,self._token_indexers)} 
        fields["pos_tag"] = SequenceLabelField([word.tag_ for word in self._tokenizer.tokenize(text)], fields["text"], label_namespace="pos_tag")
        fields['labels'] = LabelField(label)
        return Instance(fields)

        
