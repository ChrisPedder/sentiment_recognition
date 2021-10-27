"""
Defines vectorization routines for text.
"""
import numpy as np
from typing import List

import tensorflow as tf
from transformers import (BertTokenizer, TFBertModel,
                          TFAutoModel, AutoTokenizer,
                          DistilBertTokenizer, TFDistilBertModel)

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


BERT_MODEL_PARAMS = {
    'bertweet': {'path': 'vinai/bertweet-base',
                 'tokenizer': AutoTokenizer,
                 'model': TFAutoModel},
    'distilbert': {'path': 'distilbert-base-cased',
                   'tokenizer': DistilBertTokenizer,
                   'model': TFDistilBertModel},
    'bert': {'path': 'bert-base-cased',
             'tokenizer': BertTokenizer,
             'model': TFBertModel},
}

class BERTVectorizer(object):

    """
    This guy outputs the sentence classification part of the BERT embeddings
    for each sentence. the data processing keeps e.g. casing of words, as Bert
    can use this information.
    Note that I only return this CLS-embedding and then feed straight to our
    classification layer.
    """
    def __init__(self, max_length, bert_model_type='bertweet'):
        self.max_length = max_length
        self.bert_model_type = bert_model_type
        self.tokenizer = BERT_MODEL_PARAMS[
            bert_model_type]['tokenizer'].from_pretrained(
                 BERT_MODEL_PARAMS[bert_model_type]['path'], use_fast=False
        )
        self.model = BERT_MODEL_PARAMS[
            bert_model_type]['model'].from_pretrained(
                BERT_MODEL_PARAMS[bert_model_type]['path']
        )

        self.cleaning_pipeline = TransformerPipeline
        self.dict_size = 768

    def _clean_text(self, text_list):
        if not isinstance(text_list, list):
            text_list = [text_list]
        cleaned_text = [
            self.cleaning_pipeline.process(text) for text in text_list]
        return cleaned_text

    def vectorize_batch(self, batch):
        """
        Vectorization routine to use during training of the model. This works
        on batches, and benefits from the improved efficiency of computing BERT
        embeddings in batch.
        """
        texts = batch['texts']
        labels = batch['labels']

        sentences = self._clean_text(texts)

        input_ids = tf.constant(
            self.tokenizer(
                sentences, padding=True, truncation=True
            )['input_ids']
        )
        outputs = self.model(input_ids)
        return self._get_cls_state(outputs), labels

    def vectorize_instance(self, text):
        """
        Vectorization routine for vectorizing a single instance of text. To be
        used when serving the complete model, to predict the class of a given
        user utterance.
        """
        sentence = self._clean_text(text)

        input_ids = tf.constant(
            self.tokenizer(
                sentence, padding=True, truncation=True
            )['input_ids']
        )
        outputs = self.model(input_ids)
        return self._get_cls_state(outputs)

    def _get_cls_state(self, out_tensor):
        """
        Helper function to get the encoding of the CLS token for the different
        models we allow the user to select between.
        """
        if self.bert_model_type == 'bertweet':
            cls_state = out_tensor[1]
            # The last hidden-state is the first element of the output tuple

        elif self.bert_model_type == 'distilbert':
            cls_state = out_tensor[0][:,0]

        elif self.bert_model_type == 'bert':
            cls_state = out_tensor[0][:, 0, :]

        return cls_state.numpy()


if __name__ == '__main__':
     # bow = BoWVectorizer(10)
    # bow.fit(dataset)
    # print(bow.tokens_to_index)
    # print(bow.vectorize([["the", "bat", "sat", "on", "the", "hat"]]))
    dataset = {'texts': ['the cat sat on the mat'],
               'labels': [] }
    bv = BERTVectorizer(50, bert_model_type='bert')
    print(bv.vectorize(dataset))
