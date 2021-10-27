"""
Model definition for Neural networks for predictions from the BERT embeddings.

"""
import os
import argparse
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score)

from utils import (safe_folder_create, jsonify, write_config_to_json,
    create_model_training_folder)
from tf_records import TFRecordsReader
from vectorizer import BERTVectorizer

class NeuralModel:

    def __init__(self, config):
        self.config = config
        self.tf_records_ds = TFRecordsReader(config).datasets
        self.model_path = os.path.join('models', self.config.model_type)
        safe_folder_create('models')
        safe_folder_create(self.model_path)
        self.model = self.build_model()
        self.vectorizer = BERTVectorizer(config.max_length)

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.config.embed_size,)))
        model.add(tf.keras.layers.Dense(self.config.hidden_size, activation='relu'))
        model.add(tf.keras.layers.Dropout(self.config.dropout))
        model.add(tf.keras.layers.Dense(6, activation='sigmoid'))
        return model

    def train(self):
        self.logs_folder = create_model_training_folder(self.model_path)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model.compile(optimizer='rmsprop',
                           loss=loss_fn,
                           metrics=['accuracy'])

        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self.logs_folder, 'checkpoints'),
            monitor='val_accuracy',
            verbose=1, save_best_only=True,
            save_weights_only=True)

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir = os.path.join(self.logs_folder, 'tensorboard'),
            histogram_freq = 1,
            write_graph = True)

        self.model.fit(
            x=self.tf_records_ds['train'],
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=self.tf_records_ds['dev'],
            callbacks = [checkpointer, tensorboard])

        write_config_to_json(args, self.logs_folder)
        print(f"Config written to {self.logs_folder}")
        self._save_model()
        print(f"Model saved to {self.logs_folder}")

    def _save_model(self):
        latest = tf.train.latest_checkpoint(self.logs_folder)
        self.model.load_weights(latest)
        save_path = os.path.join(self.logs_folder, 'best_model')
        self.model.save(save_path)
        print(f"Best model saved to {save_path}")

    def _reload_model(self, path=None):
        save_path = os.path.join(self.logs_folder, 'best_model')
        reload_path = path or save_path
        reconstructed_model = tf.keras.models.load_model(reload_path)
        print(f"Model reloaded from {reload_path}")
        return reconstructed_model

    def _predict(self, model, inputs, predict_on_text=False):
        if predict_on_text:
            batch = {'texts': inputs,
                     'labels': []}
            inputs, _ = vectorizer.vectorize(batch)
        out = model.predict(inputs)
        return out

    def evaluate(self):
        model = self._reload_model()
        # Evaluate the model
        loss, acc = model.evaluate(self.tf_records_ds['test'])
        print(f"Test set loss {loss}")
        print(f"Test set accuracy {acc}")

        y_preds = []
        y_true = []
        for batch, labels in self.tf_records_ds['test']:
            predictions = self._predict(model, batch)
            label_preds = np.argmax(predictions, axis=1)
            y_preds.extend(label_preds.tolist())
            y_true.extend(labels.numpy().tolist())

        confusion = confusion_matrix(
            y_true, y_preds, labels=[0, 1, 2, 3, 4, 5])

        # classification report for precision, recall f1-score and accuracy
        matrix = classification_report(
            y_true, y_preds, labels=[0, 1, 2, 3, 4, 5])
        print('Classification report : \n',matrix)

        accuracy = accuracy_score(y_true, y_preds)
        f1 = f1_score(
            y_true, y_preds, labels=[0, 1, 2, 3, 4, 5], average='macro')

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
    parser.add_argument('--embed_size', help='The size of the embedding '
                        'produced by BERT', type=int,
                        default=768)
    parser.add_argument('--dataset', help='Which dataset to read from '
                        'tfrecords',
                        type=str, required=True,
                        choices=['sentiments', 'emotions'])
    parser.add_argument('--model_type', help='What sort of neural model to use ',
                        type=str, default='MLP')
    parser.add_argument('--epochs', help='How many epochs to train the model '
                        'for.', type=int, default=50)
    parser.add_argument('--dropout', help='How much dropout to apply to model ',
                        type=float, default=0.5)
    parser.add_argument('--log_dir', help='Where to save model weights and '
                        'config.', type=str, required=True)
    parser.add_argument('--hidden_size', help='What hidden sizes to use in '
                        'model.', type=int, default=256)
    parser.add_argument('--learning_rate', help='What learning rate to use in '
                        'training the model.', type=float, default=0.0001)
    parser.add_argument('--max_length', help='Maximum length of sentences in '
                        'words when tokeninizing', type=int, default=50)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    model = NeuralModel(args)
    model.train()
    model_perf = model.evaluate()
    print(f"Model performance and parameters were: {model_perf}")
