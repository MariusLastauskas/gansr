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

train_data = gan_io.load_list('./prep_data/train.dat')
train_mini = gan_io.load_list('./prep_data/train_mini.dat')

BUFFER_SIZE = 60000
BATCH_SIZE = 256

input_neurons = 60 * 60 * 4

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_mini)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

######## Main ########
# inst models
generator = nn.make_generator_model(60 * 60 * 4)
discriminator = nn.make_discriminator_model([120, 120, 4])

######## Define optimizers #######
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


######## Define checkpoints ########
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

####### Define training ########
EPOCHS = 100
num_examples_to_generate = 1

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
preview_input = train_mini[0:num_examples_to_generate]

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    print('################# train_step')

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        print(images[0])
        print(images[1])
        # print(tf.reshape(images[1], [len(images[1]), input_neurons]))
        generated_images = generator(tf.reshape(images[1], [len(images[1]), input_neurons]), training=True)
        print('################# generated_images')
        print(generated_images)
        # print(tf.image.central_crop(generated_images, 0.9375))
        print(tf.reshape(generated_images, [len(generated_images), 128, 128, 4]))
        generated_images = tf.reshape(generated_images, [len(generated_images), 128, 128, 4])
        generated_images = tf.image.central_crop(generated_images, 0.9375)

        print('################# discrimination')
        print(images[0])
        print(generated_images)
        real_output = discriminator(images[0], training=True)
        fake_output = discriminator(generated_images, training=True)
        print('################# fake_output discrimination')
        print(fake_output)

        gen_loss = nn.generator_loss(fake_output)
        disc_loss = nn.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print('################# train_step done')

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        display.clear_output(wait=True)
        print('#################3 start generate and save images')
        gan_io.generate_and_save_images(generator,
                                epoch + 1,
                                tf.reshape(preview_input, [num_examples_to_generate, input_neurons]))

        print('################### generate and save')
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    gan_io.generate_and_save_images(generator,
                            epochs,
                            tf.reshape(preview_input, [num_examples_to_generate, input_neurons]))

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(train_dataset, EPOCHS)