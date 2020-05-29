# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:43:35 2020

@author: T530
"""
import logging
import os
import sys
import csv
import pandas as pd
import numpy as np
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        
        if all([len(label)==1 for label in example.labels]):
            label_ids = label_map[example.labels[0]]
        else:
            label_ids = [0]*len(label_list)
            for label in example.labels:
                if label != '':
                    label_id = label_map[label]
                    label_ids[label_id] = 1
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            
            logger.info("labels: %s" % " ".join([str(x) for x in labels]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class DataProcessor():
    """Processor for the Frames data set (Wiki_70k version)."""

    def get_train_examples(self, data_path):
        """See base class."""
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(
            self._read_tsv(data_path), "train")

    def get_dev_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path), "dev")

    def get_labels(self, train_path, dev_path):
        """See base class."""
        train_examples = self.get_train_examples(train_path)
        dev_examples = self.get_dev_examples(dev_path)
        
        labels_2d = [i.labels for i in train_examples] + [i.labels for i in dev_examples]
        labels_2d = [i for i in labels_2d if i!=[""]]
        return sorted(list(set([j for sub in labels_2d for j in sub])))
    
    def _create_examples(self, df, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        lines = df.to_dict(orient='records')
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            sentence = line["data"]
            labels = line["labels"]
            if str(labels)=="nan":
                labels=""
            examples.append(
                InputExample(guid=guid, text_a=sentence, labels=str(labels).split(',')))
        return examples

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return pd.read_csv(input_file, delimiter='\t')
dp = DataProcessor()

model_ = "bert"
bert_model = "bert-base-uncased"
xlnet_model = "xlnet-base-cased"
train_examples = dp.get_train_examples("train.tsv")
dev_examples = dp.get_dev_examples("dev.tsv")    
labels = dp.get_labels("train.tsv", "dev.tsv")
tokenizers = {
    "bert": BertTokenizer.from_pretrained(bert_model, do_lower_case=True),
    "xlnet": XLNetTokenizer.from_pretrained(xlnet_model, do_lower_case=True),
}


tokenizer = tokenizers[model_]
train_features = convert_examples_to_features(
            train_examples, labels, 128, tokenizer)

if all([len(label)==1 for label in train_examples.labels]):
    models = {
        "bert": BertForSequenceClassification.from_pretrained(bert_model),
        "xlnet": XLNetForSequenceClassification.from_pretrained(xlnet_model),
        }
else:
    models = {
        "bert": BertForMultiLabelSequenceClassification.from_pretrained(bert_model, num_labels=len(labels)),
        "xlnet": XLNetForMultiLabelSequenceClassification.from_pretrained(xlnet_model, num_labels=len(labels)),
        }