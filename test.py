# coding: utf-8

from __future__ import print_function

from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation, merge
from keras.layers.merge import Concatenate
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from keras.utils.np_utils import to_categorical

import numpy as np
import scipy.io as sio
import csv


def load_1006210(file_name="SR99_1006210_flow-sequence_padded.mat"):
    data = sio.loadmat(file_name)
    data_x = data['padded_seq']
    data_y = data['y_seq']
    max_xy = data['max']
    min_xy = data['min']

    return data_x, data_y, max_xy, min_xy


def build_generator():
    # we will map time series x to label y (usually same with x series)
    lstm = Sequential()

    lstm.add(LSTM(6, input_shape=(6, 1), return_sequences=True))
    lstm.add(LSTM(6, return_sequences=True))
    lstm.add(LSTM(6, return_sequences=False))
    lstm.add(Dense(1, activation='tanh'))

    # this is the z space commonly refer to in GAN papers
    x = Input(shape=(6, 1))
    fake_y = lstm(x)

    return Model(x, fake_y)


def get_confusion(p, y):
    confusion = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(0, p.shape[0]):
        if np.array_equal(p[i, :], np.array([1, 0, 0])):
            if np.array_equal(y[i, :], np.array([1, 0, 0])):
                confusion[0, 0] += 1
            elif np.array_equal(y[i, :], np.array([0, 1, 0])):
                confusion[0, 1] += 1
            elif np.array_equal(y[i, :], np.array([0, 0, 1])):
                confusion[0, 2] += 1
        if np.array_equal(p[i, :], np.array([0, 1, 0])):
            if np.array_equal(y[i, :], np.array([1, 0, 0])):
                confusion[1, 0] += 1
            elif np.array_equal(y[i, :], np.array([0, 1, 0])):
                confusion[1, 1] += 1
            elif np.array_equal(y[i, :], np.array([0, 0, 1])):
                confusion[1, 2] += 1
        if np.array_equal(p[i, :], np.array([0, 0, 1])):
            if np.array_equal(y[i, :], np.array([1, 0, 0])):
                confusion[2, 0] += 1
            elif np.array_equal(y[i, :], np.array([0, 1, 0])):
                confusion[2, 1] += 1
            elif np.array_equal(y[i, :], np.array([0, 0, 1])):
                confusion[2, 2] += 1
    return confusion


def build_discriminator():
    x = Input(shape=(6, 1))
    y = Input(shape=(1,))

    yy = Reshape((1, 1))(y)

    xx = Reshape((6, 1))(x)

    xy = merge([xx, yy], mode="concat", concat_axis=1)

    lstm = Sequential()
    lstm.add(LSTM(7, input_shape=(7, 1), return_sequences=True))
    lstm.add(LSTM(6 + 1, return_sequences=True))
    lstm.add(LSTM(6 + 1, return_sequences=False))

    features = lstm(xy)
    fake = Dense(1, activation='tanh', name='generation')(features)

    return Model([x, y], fake)


if __name__ == '__main__':

    np.random.seed(1337)

    epochs = 500
    batch_size = 100
    instance = 1

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.002
    adam_beta_1 = 0.5

    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=['binary_crossentropy'])

    # build the generator
    generator = build_generator()
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss=['mse'])
    generator.load_weights('lstm_regression_params_generator_epoch_434.hdf5')

    x = Input(shape=(6, 1))
    y = Input(shape=(1,))

    # get a fake image
    fake_y = generator(x)

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake = discriminator([x, fake_y])
    combined = Model(x, fake)

    combined.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                     loss=['binary_crossentropy']
                     )

    # train data
    data_x, data_y, max_xy, min_xy = load_1006210()
    data_x = 2 * (data_x.astype(np.float32) - min_xy) / (max_xy - min_xy) - 1
    data_y = 2 * (data_y.astype(np.float32) - min_xy) / (max_xy - min_xy) - 1
    train_num = 15000
    X_train = data_x[0:train_num]
    y_train = data_y[0:train_num]
    X_test = data_x[train_num:]
    y_test = data_y[train_num:]
    test_num = X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    generated_history = y_test

    for epoch in range(epochs):

        num_batches = int(train_num / batch_size)
        progress_bar = Progbar(target=num_batches)

        epoch_com_loss = []
        epoch_disc_loss = []

        print('Epoch {} of {}'.format(epoch + 1, epochs))

        for index in range(num_batches):
            progress_bar.update(index)

            # get a batch of real data
            x_batch = X_train[index * batch_size:(index + 1) * batch_size]
            y_batch = y_train[index * batch_size:(index + 1) * batch_size]

            generated_y = generator.predict(x_batch, batch_size=100, verbose=0)

            X = np.concatenate((x_batch, x_batch))
            Y = np.concatenate((y_batch, generated_y))
            y = np.array([1] * batch_size + [0] * batch_size)
            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch([X, Y], y))

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * batch_size)

            epoch_com_loss.append(combined.train_on_batch(X, trick))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        generator_test_loss = generator.evaluate(X_test, y_test, verbose=False)

        generated_y = generator.predict(X_test, batch_size=100, verbose=0)
        generated_history = np.concatenate((generated_history, generated_y), axis=1)

        X = np.concatenate((X_test, X_test))
        Y = np.concatenate((y_test, generated_y))
        y = np.array([1] * test_num + [0] * test_num)

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate([X, Y], y, verbose=False)
        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        trick = np.ones(2 * test_num)

        combined_test_loss = combined.evaluate(X, trick, verbose=False)
        combined_train_loss = np.mean(np.array(epoch_com_loss), axis=0)

        train_history['combined'].append(combined_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['combined'].append(combined_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.5f}'
        print(ROW_FMT.format('generator (test)',
                             test_history['generator'][-1]))
        print(ROW_FMT.format('combined (train)',
                             train_history['combined'][-1]))
        print(ROW_FMT.format('combined (test)',
                             test_history['combined'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)
    pickle.dump({'train': train_history, 'test': test_history}, open('lstm_gan_predict-history6.pkl', 'wb'))