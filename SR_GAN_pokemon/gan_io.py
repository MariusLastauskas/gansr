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

def save_list(list, file_path, normalization_factor = 1):
    with open(file_path, 'w') as writer:
        for item in list:
            writer.write('[')
            for vector in item:
                writer.write('[')
                for pixel in vector:
                    writer.write('[')
                    for rgba in pixel:
                        writer.write('%f ' % (rgba / normalization_factor))
                    writer.write(']')
                writer.write(']')
            writer.write(']\n')

def load_list(file_path):
    kk = 0
    list = []
    with open(file_path, 'r') as reader:
        for item in reader:
            print('item')
            image = []
            for v in item.split(']][['):
                vector = []
                for p in v.split(']['):
                    pixel = []
                    for rgba in p.replace('[', '').replace(']', '').split(' '):
                        if len(rgba) > 0 and rgba != "\n":                        
                            pixel.append(float(rgba))
                    vector.append(pixel)
                image.append(vector)
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