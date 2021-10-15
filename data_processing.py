"""
Class to read and batch data from the csv file and pass to the model.
"""

import csv
import ast
import numpy as np
import pandas as pd
from io import StringIO

# helper function to parse byte-encoded data
def parse_bytes(field):
    result = field
    try:
        result = ast.literal_eval(field)
    finally:
        return result.decode() if isinstance(result, bytes) else field

class DataIterator:

    def __init__(self, batch_size, dataset, dataset_split):
        self.dataset_path = f'data/{dataset}/{dataset_split}.csv'
        self.batch_size = batch_size
        self.sentiment_map = {
            "0": "negative",
            "4": "positive",
        }

    def _process_line(self, line):
        text, sentiment = parse_bytes(line[5]), line[0]
        obj = {"text": text,
               "sentiment": self.sentiment_map[sentiment]}
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
        sentiments = []
        for obj in batch:
            texts.append(obj['text'])
            sentiments.append(obj['sentiment'])
        return {
            'texts': texts,
            'sentiments': sentiments
        }

if __name__ == "__main__":
    iterator = DataIterator(3, 'test').batched()
    for i in range(3):
        print(next(iterator))
