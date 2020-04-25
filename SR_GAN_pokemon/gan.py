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








image = imageio.imread('./train_images/abra.png')
print(image[60][60])

plt.imshow(image)
plt.show()

####################### prepare train image data
# train_images = []
# kk = 0

# for file in os.listdir(train_images_path):
#     image = imageio.imread('./train_images/' + file)
#     data_list = []
#     for i in range(len(image)):
#         data_vector = []
#         for j in range(len(image[0])):
#             rgba_vector = []
#             for k in range(len(image[0][0])):
#                 rgba_vector.append((image[i][j][k] - 127.5) / 127.5)
#             data_vector.append(rgba_vector)
#         data_list.append(data_vector)
#     train_images.append(data_list)
#     kk = kk + 1
#     print(kk)

# gan_io.save_list(train_images, train_images_file)


####################### prepare train image data
# train_mini_images = []
# kk = 0

# for file in os.listdir(train_images_path):
#     image = imtr.image_minify(imageio.imread('./train_images/' + file))
#     data_list = []
#     for i in range(len(image)):
#         data_vector = []
#         for j in range(len(image[0])):
#             rgba_vector = []
#             for k in range(len(image[0][0])):
#                 rgba_vector.append((image[i][j][k] - 127.5) / 127.5)
#             data_vector.append(rgba_vector)
#         data_list.append(data_vector)
#     train_mini_images.append(data_list)
#     kk = kk + 1
#     print(kk)
    
# gan_io.save_list(train_mini_images, train_mini_images_file)

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data('fashion')
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
# train_images = imtr.filter_data(train_images, train_labels, 4)

# k = 0
# mini_train_images_file = './mini.dat'

# print('####################### Saving minified images')
# mini_train_images = []
# train_images = [image]

# for image in train_images:
#     mini = imtr.image_minify(image)
#     for i in range(len(mini)):
#         for j in range(len(mini[0])):
#             for k in range(len(mini[0][0])):
#                 mini[i][j][k] = (mini[i][j][k] - 127.5) / 127.5

#     mini_train_images.append(mini)
#     print(k)
#     k = k + 1
# print('dim')
# print(len(mini))
# print(len(mini[0]))
# print(len(mini[0][0]))

# gan_io.save_list(mini_train_images, mini_train_images_file)
# print('####################### Minified images saved')

###

# print('####################### Loading images')
# l = gan_io.load_list(train_mini_images_file)
# train_images = gan_io.load_list(train_images_file)
# print('####################### Minified images loaded')

# BUFFER_SIZE = 60000
# BATCH_SIZE = 256

# # Batch and shuffle the data
# # train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# train_dataset = tf.data.Dataset.from_tensor_slices((train_images, l)).batch(BATCH_SIZE)
# print('####################### Split dataset into batches')

# ######## Main ########
# # inst models
# generator = nn.make_generator_model()
# discriminator = nn.make_discriminator_model()

# ######## Define optimizers #######
# generator_optimizer = tf.keras.optimizers.Adam(1e-4)
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ######## Define checkpoints ########
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)

# ####### Define training ########
# EPOCHS = 1000
# num_examples_to_generate = 16

# # We will reuse this seed overtime (so it's easier)
# # to visualize progress in the animated GIF)
# preview_input = l[0:num_examples_to_generate]

# # Notice the use of `tf.function`
# # This annotation causes the function to be "compiled".
# @tf.function
# def train_step(images):
#     print('################# train_step')
#     print(images)

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         print('################ image shape')
#         print(images[0].shape)
#         generated_images = generator(tf.reshape(images[1], [len(images[0]), images[0].shape[0] * images[0].shape[1] * images[0].shape[2]]), training=True)

#         real_output = discriminator(images[0], training=True)
#         fake_output = discriminator(generated_images, training=True)

#         gen_loss = nn.generator_loss(fake_output)
#         disc_loss = nn.discriminator_loss(real_output, fake_output)

#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
#     print('################# train_step done')

# def train(dataset, epochs):
#     for epoch in range(epochs):
#         start = time.time()

#         for image_batch in dataset:
#             train_step(image_batch)

#         # Produce images for the GIF as we go
#         display.clear_output(wait=True)
#         print('#################3 start generate and save images')
#         gan_io.generate_and_save_images(generator,
#                                 epoch + 1,
#                                 tf.reshape(preview_input, [num_examples_to_generate, 14*14]))

#         print('################### generate and save')
#         # Save the model every 15 epochs
#         if (epoch + 1) % 15 == 0:
#             checkpoint.save(file_prefix = checkpoint_prefix)

#         print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

#     # Generate after the final epoch
#     display.clear_output(wait=True)
#     gan_io.generate_and_save_images(generator,
#                             epochs,
#                             tf.reshape(preview_input, [num_examples_to_generate, 14*14]))

# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
# train(train_dataset, EPOCHS)