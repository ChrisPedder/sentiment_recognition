"""
A routine for writing TFRecords files of the sentence embeddings and labels
for the datasets used to train the NN & regression models.
"""

import os
import argparse
import tensorflow as tf
import numpy as np

from vectorizer import BERTVectorizer
from data_processing import (SentimentDataIterator, EmotionDataIterator)

# helper functions for converting float and int features to tf.train compatible
# features
def _floats_feature(value):
   return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordsWriter(object):
    def __init__(self, args):
        self.SEED = 42
        self.args = args
        self.batch_size = args.batch_size
        self.predictor = BERTVectorizer(args.max_length)
        self.datasets = self.get_tf_datasets()

        self.tfrecords_foldername = os.path.join(
            './data',
            args.dataset + '_bert_tfrecords')
        if not os.path.isdir(self.tfrecords_foldername):
            os.mkdir(self.tfrecords_foldername)

    def get_tf_datasets(self):
        if args.dataset == 'sentiments':
            train_ds = SentimentDataIterator(self.batch_size, 'train').batched()
            dev_ds = SentimentDataIterator(self.batch_size, 'dev').batched()
            test_ds = SentimentDataIterator(self.batch_size, 'test').batched()
        elif args.dataset == 'emotions':
            train_ds = EmotionDataIterator(self.batch_size, 'train').batched()
            dev_ds = EmotionDataIterator(self.batch_size, 'val').batched()
            test_ds = EmotionDataIterator(self.batch_size, 'test').batched()
        else:
            raise ValueError(
                f"Expected dataset in [sentiments, emotions] but got {dataset}")

        return {'train': train_ds,
                'dev': dev_ds,
                'test': test_ds}

    def write_dataset_to_tf_records(self, dataset, name):
        tfrecords_filename = os.path.join(self.tfrecords_foldername, name)
        writer = tf.io.TFRecordWriter(tfrecords_filename)
        i = 0
        while True:
            try:
                batch = next(dataset)
                print(f'Converting batch {i} to TFRecords')
                features, labels = self.predictor.vectorize(batch)

                for embedding, label in zip(features, labels):

                    embed_tensor = tf.io.serialize_tensor(embedding.tolist())
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'embedding': _bytes_feature(embed_tensor),
                                'labels': _int64_feature(label),
                            }
                        )
                    )

                    writer.write(example.SerializeToString())
                i += 1
            except StopIteration:
                break
        writer.close()
        print(f"Writing for {name} finished")

    def write_to_tf_records(self):
        for name, dataset in self.datasets.items():
            self.write_dataset_to_tf_records(dataset, name)
        print("All data written to tfrecords")



class TFRecordsReader(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.tfrecords_foldername = os.path.join(
            './data',
            args.dataset + '_bert_tfrecords')
        self.datasets = self.read_from_tf_records()

    def read_from_tf_records(self):
        dataset_dict = {}
        for name in ['train', 'dev', 'test']:
            raw_dataset = tf.data.TFRecordDataset(
                os.path.join(self.tfrecords_foldername, name))
            parsed_dataset = raw_dataset.map(
                self._parse_example_function)
            dataset_dict[name] = parsed_dataset.batch(
                self.batch_size)

        return dataset_dict

    def _parse_example_function(self, example_proto):
        indices = [0, 1, 2, 3, 4, 5]
        # Create a dictionary describing the features.
        image_feature_description = {
            'embedding': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.FixedLenFeature([], tf.int64),
        }
        # Parse the input tf.train.Example proto using the dictionary above.
        example = tf.io.parse_single_example(
            example_proto, image_feature_description)
        embedding = tf.io.parse_tensor(
            example['embedding'], out_type = tf.float32)
        return embedding, example['labels']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='The size of the batches to use '
                        'when writing data to TFRecords files', type=int,
                        default=128)
    parser.add_argument('--dataset', help='Which dataset to serialize to '
                        'tfrecords',
                        type=str, required=True,
                        choices=['sentiments', 'emotions'])
    parser.add_argument('--max_length', help='Path to the data files',
                        type=int, default=50)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    tfr = TFRecordsWriter(args)
    tfr.write_to_tf_records()
    print("finished")
