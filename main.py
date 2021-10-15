"""
Model training and evaluation routines.

To train the model,
* create a venv
* install the requirements `pip install -r requirements.txt`

- To train...
* python main.py  \
    --model_type MODEL_TYPE \

- To evaluate...
* python main.py \
    --model_type MODEL_TYPE \
    --job evaluate \

"""

import argparse

from data_processing import DataIterator
from vectorizer import BoWVectorizer, BERTVectorizer
from regression import RegressionModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help='The size of the batches to use '
                        'when training the models', type=int,
                        default=32)
    parser.add_argument('--max_length', help='Maximum number of tokens to '
                        'use when vectorizing text', type=int,
                        default=50)
    parser.add_argument('--model_type', help='Model type to use ',
                        type=str, default='BoW')
    parser.add_argument('--job', help='Whether to train or evaluate the model.',
                        type=str, default='train')
    args = parser.parse_args()
    return args

VECOTORIZERS = {
    'BoW': BoWVectorizer,
    'bert': BERTVectorizer
}

if __name__ == '__main__':

    args = parse_args()
    model = RegressionModel(args, DataIterator, VECOTORIZERS[args.model_type])

    if args.job == 'train':
        model.train()
    elif args.job == 'evaluate':
        results = model.evaluate()
        print(results)
