import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import argparse

from ncps.wirings import AutoNCP
from ncps.tf import LTC

from scipy.io import loadmat
import sklearn
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

class EMGData:
    def __init__(self, location):
        # Read data from MATLAB file
        print('Loading data...')
        matfile = loadmat(location);

        x = np.array(matfile['data'])
        y = np.array(matfile['labels'])
        windowSize = matfile['windowSize']
        numFeatures = matfile['numFeatures']

        print('Processing data...')

        x = np.transpose(x, (0, 2, 1))
        y = np.transpose(y)

        y[y == 10] = 2
        y[y == 12] = 3

        print('X shape: {}'.format(x.shape))
        print('Y shape: {}'.format(y.shape))

        print('Formatting data into windows...')

        batchSize = 5
        xWindows = np.zeros([x.shape[0], batchSize, x.shape[1], x.shape[2]])
        yWindows = np.zeros([y.shape[0], batchSize, y.shape[1]])

        for i in range(batchSize, x.shape[0]):
            for j in range(batchSize):
                xWindows[i - 1, j, :, :] = x[i - batchSize, :, :]
                yWindows[i - 1, j, :] = y[i - batchSize, :]

        xWindows = xWindows[batchSize - 1:, :, :, :]
        yWindows = yWindows[batchSize - 1:, :, :]

        # Resize data to correct shape
        xWindows = np.reshape(xWindows, [xWindows.shape[0], xWindows.shape[1], xWindows.shape[2], xWindows.shape[3], 1])

        print('X shape: {}'.format(xWindows.shape))
        print('Y shape: {}'.format(yWindows.shape))

        print('Making train/val/test splits...')

        # Make train, validation, test sets
        train_x, test_x, train_y, test_y = train_test_split(xWindows, yWindows, test_size=0.2, random_state=42)
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.25, random_state=42)

        self.valid_x = val_x
        self.train_x = train_x
        self.test_x = test_x
        self.valid_y = val_y
        self.train_y = train_y
        self.test_y = test_y

        print(self.train_x.shape)
        print(self.valid_x.shape)
        print(self.test_x.shape)
        print(self.train_y.shape)
        print(self.valid_y.shape)
        print(self.test_y.shape)

        print("Total number of training sequences: {}".format(self.train_x.shape[0]))
        print("Total number of validation sequences: {}".format(self.valid_x.shape[0]))
        print("Total number of test sequences: {}".format(self.test_x.shape[0]))

    def iterate_train(self, batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[permutation[start:end]]
            yield (batch_x, batch_y)

class EMGClassifierModel:
    def __init__(self, emg_data, model_type, model_size, learning_rate=0.01):
        self.height, self.width, self.channels = (emg_data.train_x.shape[2], emg_data.train_x.shape[3], emg_data.train_x.shape[4])

        self.wiring = AutoNCP(20, output_size=4)

        print('Creating model...')
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(None, self.height, self.width, self.channels)),
                tf.keras.layers.Normalization(),
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv2D(64, (3, 3))
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Conv2D(32, (3, 3))
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D()),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32)),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.2),
                LTC(self.wiring, return_sequences=True),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("softmax")),
            ]
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
            loss='sparse_categorical_crossentropy',
            metrics='accuracy'
        )

        self.model.summary(line_length=100)

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)

    def fit(self, emg_data, epochs, checkpoint_filepath='/tmp/checkpoint', verbose=True, log_period=50):
        self.checkpoint_filepath = checkpoint_filepath

        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        self.es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=10)

        self.lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(self.scheduler)

        self.history = self.model.fit(
            x=emg_data.train_x, y=emg_data.train_y, batch_size=32, epochs=epochs,
            validation_data=(emg_data.valid_x, emg_data.valid_y),
            callbacks = [self.model_checkpoint_callback, self.es_callback, self.lr_scheduler_callback]
        )

        self.model.load_weights(checkpoint_filepath)

    def scheduler(self, epoch, lr):
        if epoch % 50 == 0:
            return lr * 0.5
        else:
            return lr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="lstm")
    parser.add_argument('--log', default=1, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    args = parser.parse_args()

    print(tf.config.list_physical_devices('GPU'))

    emg_data = EMGData('Data\TrainingData_20230314-112821_113353_SJ_Pilot_03_14_10-Aug-2023_12-29-43.mat')
    classifier = EMGClassifierModel(emg_data, model_type=args.model, model_size=args.size)

    classifier.fit(emg_data, epochs=args.epochs, log_period=args.log)
