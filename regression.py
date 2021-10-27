"""
Model definition for regression based on the BoW model.

"""
import os
import argparse
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score)
from joblib import dump, load

from data_processing import DataIterator
from vectorizer import BoWVectorizer, BERTVectorizer
from regression import RegressionModel
from utils import safe_folder_create


VECTORIZERS = {
    'BoW': BoWVectorizer,
    'bert': BERTVectorizer
}

class RegressionModel:

    ONE_HOT_DICT = {
        'positive': 1,
        'neutral': 0,
        'negative': -1
    }

    def __init__(self, config, dataset, vectorizer):
        self.config = config
        self.dataset = dataset
        self.vectorizer = self.get_vectorizer(vectorizer)
        self.model_path = os.path.join('models', self.config.model_type)
        safe_folder_create('models')
        safe_folder_create(self.model_path)

    def get_vectorizer(self, vectorizer):
        if self.config.model_type == "BoW":
            train = self.dataset(
                self.config.batch_size, 'sentiment_1400k', 'train').batched()
            vectorizer = vectorizer(self.config.max_length).fit(
                train)
        elif self.config.model_type == "bert":
            vectorizer = vectorizer(self.config.max_length)

        return vectorizer

    def _vectorize_batch(self, batch):
        data_out, labels = self.vectorizer.vectorize(batch)
        return data_out, labels

    def train(self, _C=1.0):
        print("Creating new model")
        model = SGDClassifier(loss="log", max_iter=4000, )
        # model = LogisticRegression(max_iter = 4000, C=_C)
        classes = np.unique(["positive", "negative"])

        for i, batch in enumerate(
            self.dataset(
                self.config.batch_size, 'sentiment_1400k', 'train').batched()):

            x_tr, y_tr = self._vectorize_batch(batch)
            model.partial_fit(x_tr, y_tr, classes=classes)

            if i % 100 == 0:
                print(f"Batch {i} processed.")
        print("Model trained")
        self._save_model(model, 'sentiment_model')

    def _save_model(self, model, model_name):
        model_path = self.model_path + f'/{model_name}.joblib'
        dump(model, model_path)
        print(f"Model saved to {model_path}")

    def _reload_model(self, model_name):
        clf = SGDClassifier()
        model_path = self.model_path + f'/{model_name}.joblib'
        try:
            clf = load(model_path)
            print(f"Model loaded from {model_path}")
        except NotFittedError:
            print("Need to train a model first")

        return clf

    def evaluate(self):
        model = self._reload_model('sentiment_model')

        y_preds = []
        y_actual = []
        for batch in self.dataset(
            self.config.batch_size, 'sentiment_1400k', 'test').batched():
            new_X, new_y = self._vectorize_batch(batch)

            y_preds += model.predict(new_X).tolist()
            y_actual += new_y

        confusion = confusion_matrix(
            y_actual, y_preds, labels=["positive", "negative"])

        # classification report for precision, recall f1-score and accuracy
        matrix = classification_report(
            y_actual, y_preds, labels=["positive", "negative"])
        print('Classification report : \n',matrix)

        accuracy = accuracy_score(y_actual, y_preds)
        f1 = f1_score(
            y_actual, y_preds, labels=["positive", "negative"], average='macro')

        return {
            "Model_name": self.config.model_type,
            "checkpoint_path": self.model_path,
            "accuracy": accuracy,
            "f1_score": f1
        }


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


if __name__ == '__main__':

    args = parse_args()
    iterator = DataIterator
    vectorizer = VECTORIZERS[args.model_type]
    model = RegressionModel(args, iterator, vectorizer)

    if args.job == 'train':
        model.train()
    elif args.job == 'evaluate':
        results = model.evaluate()
        print(results)
