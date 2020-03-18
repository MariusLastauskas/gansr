from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import gan_io

from IPython import display

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(14*14,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    # assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# def load_list(file_path):
#     list = []
#     with open(file_path, 'r') as reader:
#         for item in reader:
#             image = []
#             for vector in item[1:-2].split('], ['):
#                 v = []
#                 for num in vector.replace('[', '').replace(']', '').split(', '):
#                     v.append(float(num))
#                 image.append(v)
#             list.append(image)
#     return list

def image_minify(image, mask = False):
    # print('**** im *****')
    # print(image)
    if mask != False:
        # minified = tf.boolean_mask(image, mask)
        # minified = []
        # for i in range(14):
        #   minified.append(tf.boolean_mask(image[i * 2], mask))
        # minified = tf.reshape(minified, [14, 14, 1])
        # return minified
        minified = tf.boolean_mask(image, mask, axis=1)
        minified = tf.boolean_mask(minified, mask)
        minified = tf.reshape(minified, [14, 14, 1])
        return minified

    minified = []
    for i in range(28):
        if i % 2 == 0:
            vector = []
            for j in range(28):
                if j % 2 == 0:
                    vector.append(image[i][j])
            minified.append(vector)
    # print('**** min *****')
    # print(tf.stack(minified))
    return minified

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# mini_train_images_file = './mini.dat'
# l = gan_io.load_list(mini_train_images_file)

for ii in range(len(test_images)):
    if test_labels[ii] == 3:
        mini = image_minify(test_images[ii])

        for i in range(len(mini)):
            for j in range(len(mini)):
                mini[i][j] = (mini[i][j] - 127.5) / 127.5

        test = np.array(mini).reshape([1, 14*14])

        result = generator(test)

        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(tf.reshape(test, [14, 14]), cmap='gray')
        f.add_subplot(1,2, 2)
        plt.imshow(tf.reshape(result[0], [28, 28]), cmap='gray')
        plt.show(block=True)