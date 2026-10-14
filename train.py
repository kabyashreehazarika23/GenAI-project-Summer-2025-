import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
import tensorflow as tf

from utils.data_loader import DataLoader
from models.generator import build_generator
from models.discriminator import build_discriminator

# Parameters
img_rows, img_cols = 64, 64
channels = 3
img_shape = (img_rows, img_cols, channels)
epochs = 3
batch_size = 4

# Load models
generator = build_generator()
discriminator = build_discriminator(img_shape)

# Optimizers
optimizer = Adam(0.0002, 0.5)

# Compile discriminator
discriminator.compile(loss=BinaryCrossentropy(),
                      optimizer=optimizer,
                      metrics=['accuracy'])

# Define adversarial model
input_landmark = tf.keras.Input(shape=img_shape)
generated_image = generator(input_landmark)
discriminator.trainable = False
validity = discriminator([input_landmark, generated_image])

combined = tf.keras.Model(input_landmark, [validity, generated_image])
combined.compile(loss=[BinaryCrossentropy(), MeanAbsoluteError()],
                 loss_weights=[1, 100],
                 optimizer=optimizer)

# Data loader
dataloader = DataLoader('./dataset', img_res=(img_rows, img_cols))

# Training
for epoch in range(epochs):
    print(f"\n--- Starting Epoch {epoch+1}/{epochs} ---")
    num_batches = len(dataloader) // batch_size

    for batch_i in range(num_batches):
        landmarks_batch = []
        real_images_batch = []

        for idx in range(batch_i * batch_size, (batch_i + 1) * batch_size):
            landmark, img = dataloader[idx]
            landmarks_batch.append(landmark)
            real_images_batch.append(img)

        landmarks_batch = np.array(landmarks_batch)
        real_images_batch = np.array(real_images_batch)

        # Generate fake images
        fake_images = generator.predict(landmarks_batch)

        # Labels
        valid = np.ones((batch_size, 8, 8, 1))  # PatchGAN output size
        fake = np.zeros((batch_size, 8, 8, 1))

        # Train discriminator
        d_loss_real = discriminator.train_on_batch([landmarks_batch, real_images_batch], valid)
        d_loss_fake = discriminator.train_on_batch([landmarks_batch, fake_images], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        g_loss = combined.train_on_batch(landmarks_batch, [valid, real_images_batch])

        # Print progress
        print(f"[Batch {batch_i+1}/{num_batches}] D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}% | G loss: {g_loss[0]:.4f}")

    # Save models at each epoch if needed
    generator.save(f'models/saved_model/generator_epoch{epoch+1}.h5')
    discriminator.save(f'models/saved_model/discriminator_epoch{epoch+1}.h5')
