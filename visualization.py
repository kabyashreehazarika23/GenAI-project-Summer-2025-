# gan_authentication.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Generator
def build_generator(input_dim, output_dim):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_dim, activation='tanh')
    ])
    return model

# 2. Discriminator
def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# 3. Loss & Optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 4. Training Function with loss tracking
gen_losses = []
disc_losses = []
disc_accuracies = []

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

    # Calculate discriminator accuracy
    real_preds = tf.cast(real_output > 0.5, tf.float32)
    fake_preds = tf.cast(fake_output < 0.5, tf.float32)
    accuracy = (tf.reduce_mean(real_preds) + tf.reduce_mean(fake_preds)) / 2.0

    return gen_loss, disc_loss, accuracy

# 5. Initialize
real_signals = np.random.normal(size=(1000, 32))
noise_dim = 16
batch_size = 32
epochs = 1000

generator = build_generator(noise_dim, 32)
discriminator = build_discriminator(32)

# 6. Training Loop
for epoch in range(epochs):
    idx = np.random.randint(0, real_signals.shape[0], batch_size)
    real_batch = real_signals[idx]
    g_loss, d_loss, acc = train_step(real_batch)

    gen_losses.append(g_loss.numpy())
    disc_losses.append(d_loss.numpy())
    disc_accuracies.append(acc.numpy())

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Gen Loss: {g_loss.numpy():.4f} | Disc Loss: {d_loss.numpy():.4f} | Disc Acc: {acc.numpy():.4f}")

# 7. Plot Losses
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(gen_losses, label="Generator Loss")
plt.plot(disc_losses, label="Discriminator Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 8. Plot Discriminator Accuracy
plt.subplot(1, 2, 2)
plt.plot(disc_accuracies, label="Discriminator Accuracy", color='green')
plt.title("Discriminator Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# 9. Plot Real vs Fake Signals (First 5 examples)
fake_samples = generator(tf.random.normal([5, noise_dim]), training=False).numpy()
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.plot(real_signals[i], label=f"Real {i+1}", linestyle='--')
    plt.plot(fake_samples[i], label=f"Fake {i+1}")
plt.title("Real vs Fake Signals")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# 10. Save Models
os.makedirs("saved_models", exist_ok=True)
generator.save("saved_models/generator.h5")
discriminator.save("saved_models/discriminator.h5")
print("✅ Models saved to 'saved_models/'")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import seaborn as sns

# 1. Prepare real and fake signals for evaluation
num_test = 200
real_test = real_signals[:num_test]
noise = tf.random.normal([num_test, noise_dim])
fake_test = generator(noise, training=False).numpy()

X_test = np.concatenate([real_test, fake_test], axis=0)
y_true = np.array([1]*num_test + [0]*num_test)

# 2. Discriminator predictions
y_pred_prob = discriminator(X_test, training=False).numpy().flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

# 3. Evaluate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\n🎯 Discriminator Accuracy on Test Set: {accuracy*100:.2f}%")

# 4. Visualize with PCA or t-SNE
# Option 1: PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)

# Option 2: t-SNE (uncomment if needed)
# tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
# X_tsne = tsne.fit_transform(X_test)

# 5. Plot using Seaborn
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_true, palette=['red', 'blue'], style=y_pred, s=60)
plt.title("PCA: Real (1-blue) vs Fake (0-red) Signals")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(title='True Label', loc='best')
plt.grid(True)
plt.show()

