"""
Defines vectorization routines for text.
"""
import numpy as np
from typing import List

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, TFAutoModel, AutoTokenizer

from cleaning import LogisticPipeline, TransformerPipeline

class BoWVectorizer:

    def __init__(self, max_length):
        self.max_length = max_length
        self.tokens_to_index = {}
        self.dict_size = 0
        self.cleaning_pipeline = LogisticPipeline

    def _prepare_batch(self, batch):
        texts = batch['texts']
        labels = batch['labels']
        cleaned_text = [self.cleaning_pipeline.process(text) for text in texts]
        return cleaned_text, labels

    def fit(self, dataset, min_occurrences=5):
        # We get a lot of words from tweets, so set a threshold for how many
        # times a word has to appear to not be mapped to <UNK>

        counts_dict = {}
        # set the unknown char to occur enough times to be counted
        counts_dict['<UNK>'] = min_occurrences + 1

        for batch in dataset:
            cleaned_text, _ = self._prepare_batch(batch)
            for entry in cleaned_text:
                entry = self._crop_entry(entry)
                for token in entry:
                    counts_dict[token] = counts_dict.get(token, 0) + 1

        i = 0
        for k, v in counts_dict.items():
            if v >= min_occurrences:
                self.tokens_to_index[k] = i
                i += 1

        self.dict_size = len(self.tokens_to_index) + 1

        print(f"Fitted BoW vectorizer on dataset, dictionary size is \
            {self.dict_size}.")

        return self

    def _crop_entry(self, entry):
        """
        Crops entries which are longer than self.max_length to self.max_length
        """
        if len(entry) > self.max_length:
            entry = entry[:self.max_length]
        return entry

    def vectorize(self, batch):
        """
        Takes a batch entry (a dict) and returns the vectorization of the
        text data in that batch
        """
        output = np.zeros((len(batch['texts']), self.dict_size))
        cleaned_text_list, labels = self._prepare_batch(batch)
        for i, entry in enumerate(cleaned_text_list):
            entry = self._crop_entry(entry)
            for word in entry:
                if word in self.tokens_to_index:
                    output[i][self.tokens_to_index[word]] += 1
                else:
                    output[i][self.tokens_to_index['<UNK>']] += 1
        return output, labels


class BERTVectorizer(object):

    """
    This guy outputs the sentence classification part of the BERT embeddings
    for each sentence. the data processing keeps e.g. casing of words, as Bert
    can use this information.
    Note that I only return this CLS-embedding and then feed straight to our
    classification layer.
    """
    def __init__(self, max_length, bert_model_type="vinai/bertweet-base"):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_model_type, use_fast=False)
        self.model = TFAutoModel.from_pretrained(bert_model_type)
        self.cleaning_pipeline = TransformerPipeline
        self.dict_size = 768

    def _prepare_batch(self, batch):
        texts = batch['texts']
        labels = batch['labels']
        cleaned_text = [self.cleaning_pipeline.process(text) for text in texts]
        return cleaned_text, labels

    def vectorize(self, batch):
        sentences, labels = self._prepare_batch(batch)
        input_ids = tf.constant(
            self.tokenizer(
                sentences, padding=True, truncation=True
            )['input_ids']
        )
        outputs = self.model(input_ids)
        last_hidden_states = outputs[1]
        # The last hidden-state is the first element of the output tuple
        return last_hidden_states.numpy(), labels

if __name__ == '__main__':
     # bow = BoWVectorizer(10)
    # bow.fit(dataset)
    # print(bow.tokens_to_index)
    # print(bow.vectorize([["the", "bat", "sat", "on", "the", "hat"]]))
    dataset = ['the cat sat on the mat']
    bv = BERTVectorizer()
    print(bv.vectorize(dataset))
