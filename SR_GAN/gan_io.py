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

from IPython import display

def save_list(list, file_path):
    with open(file_path, 'w') as writer:
        for item in list:
            writer.write('%s\n' % item)

def load_list(file_path):
    list = []
    with open(file_path, 'r') as reader:
        for item in reader:
            image = []
            for vector in item[1:-2].split('], ['):
                v = []
                for num in vector.replace('[', '').replace(']', '').split(', '):
                    v.append(float(num))
                image.append(v)
            list.append(image)
    return list

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    print('################### generate and save image {}', epoch)
    predictions = model(test_input, training=False)
    print('################### generated image')

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))