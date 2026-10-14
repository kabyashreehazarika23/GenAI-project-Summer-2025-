# gan_authentication.py

# 1. Import Libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# 2. Define the Generator
def build_generator(input_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim, activation='tanh')  # Output: fake signal
    ])
    return model

# 3. Define the Discriminator
def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output: real (1) or fake (0)
    ])
    return model

# 4. Loss Functions
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# 5. Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 6. Training Step Function
@tf.function
def train_step(real_signals):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_signals = generator(noise, training=True)

        real_output = discriminator(real_signals, training=True)
        fake_output = discriminator(generated_signals, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# 7. Initialize Data and Models
real_signals = np.random.normal(size=(1000, 32))  # Simulated trusted signals
noise_dim = 16
batch_size = 32

generator = build_generator(noise_dim, 32)
discriminator = build_discriminator(32)

# 8. Training Loop
epochs = 1000

for epoch in range(epochs):
    idx = np.random.randint(0, real_signals.shape[0], batch_size)
    train_step(real_signals[idx])
    if epoch % 100 == 0:
        print(f"Epoch {epoch} completed")
