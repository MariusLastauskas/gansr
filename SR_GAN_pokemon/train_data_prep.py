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
import gan_image_transformations as imtr
import gan_nn as nn
import imageio

from IPython import display

train_images_path = './train_images/'
train_images_file = './train_images.dat'
train_mini_images_file = './train_mini_images.dat'

image1 = imageio.imread('./train_images/abra.png')
image2 = imageio.imread('./train_images/absol.png')

gan_io.save_list([image1, image2], './prep_data/train.dat', 255)
list = gan_io.load_list('./prep_data/train.dat')
print(image2[60][60])

print(list[1][60][60])
print(np.array(image2).shape)
print(np.array(list[1]).shape)
plt.imshow(list[1])
plt.show()