"""
Class to read and batch data from the csv file and pass to the model.
"""

import csv
import ast
import numpy as np
import pandas as pd
from io import StringIO

from abc import ABC, abstractmethod

CATEGORY_MAPS = {
    'sentiments': {'0': 0, '4': 1}, # Two classes, 0 is negative, 4 is positive
    'emotions':  {
        'anger': 0, 'sadness': 1, 'surprise': 2, 'joy': 3, 'fear': 4, 'love': 5}
    # Six classes, labelled by name
}

# helper function to parse byte-encoded data
def parse_bytes(field):
    result = field
    try:
        result = ast.literal_eval(field)
    finally:
        return result.decode() if isinstance(result, bytes) else field

class DataIterator:

    def __init__(self, batch_size):
        self.batch_size = batch_size

    @abstractmethod
    def get_iter(self):
        """
        Gets an iterator which will iterate through the csv, returning data
        line by line.
        """
        pass

    def batched(self):
        """
        Returns a batched version of the data, of size batch_size
        """
        _batch_size = getattr(self, 'batch_size')
        iterator = self.get_iter()
        while True:
            try:
                yield self._as_batch(
                    [next(iterator) for _ in range(_batch_size)]
                )
            except StopIteration:
                break

    def _as_batch(self, batch):
        """
        Helper function to batch data
        """
        texts = []
        labels = []
        for obj in batch:
            texts.append(obj['text'])
            labels.append(obj['label'])
        return {
            'texts': texts,
            'labels': labels
        }

class SentimentDataIterator(DataIterator):

    def __init__(self, batch_size, dataset_split):
        super().__init__(batch_size)
        self.dataset_path = f'data/sentiment_1400k/{dataset_split}.csv'
        self.sentiment_map = {
            "0": "negative",
            "4": "positive",
        }

    def _process_line(self, line):
        text, sentiment = parse_bytes(line[5]), line[0]
        obj = {"text": text,
               "label": self.sentiment_map[sentiment]}
        return obj

    def get_iter(self):
        """
        Gets an iterator which will iterate through the csv, returning data
        line by line.
        This was a real pain due to the format of the strings in the twitter
        data, but it's the most important part of the pipe!
        """
        with open(self.dataset_path, 'rt', encoding="ISO8859") as f:
            reader = csv.reader(f)
            for line in reader:
                yield self._process_line(line)


class EmotionDataIterator(DataIterator):

    def __init__(self, batch_size, dataset_split):
        super().__init__(batch_size)
        self.dataset_path = f'data/emotion/{dataset_split}.txt'
        self.emotion_map = CATEGORY_MAPS['emotions']

    def _process_line(self, line):
        text, emotion = line.strip().split(';')
        obj = {"text": text,
               "label": self.emotion_map[emotion]}
        return obj

    def get_iter(self):
        """
        Gets an iterator which will iterate through the csv, returning data
        line by line.
        This was a real pain due to the format of the strings in the twitter
        data, but it's the most important part of the pipe!
        """
        with open(self.dataset_path, 'r', encoding="utf8") as f:
            for line in f:
                yield self._process_line(line)


if __name__ == "__main__":
    s_iterator = SentimentDataIterator(3, 'test').batched()
    for i in range(3):
        print(next(s_iterator))

    e_iterator = EmotionDataIterator(3, 'test').batched()
    for i in range(3):
        print(next(e_iterator))
