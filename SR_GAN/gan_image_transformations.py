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

def filter_data(data_list, label_list, label):
    list = []
    for i in range(len(data_list)):
        if label_list[i] == label:
            list.append(data_list[i])
    return list

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

def image_batch_minify(images):
    m = False
    mask = []
    for i in range(28):
        mask.append(m)
        m = not m
    print('*** images.shape ***')
    print(images.shape[0])
    minified2 = []
    for i in range(images.shape[0]):
        minified2.append(image_minify(images[i], mask))
        print(i)
        # minified2.append(image_minify(image))
    print('**** minified *****')
    print(len(minified2))
    return tf.stack(minified2)