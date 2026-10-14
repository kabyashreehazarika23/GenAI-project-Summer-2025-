import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from random import shuffle

# -----------------------------
# Image loader (32x32)
# -----------------------------
def load_images(folder, size=(32,32)):
    files = glob(os.path.join(folder, '*.jpg')) + glob(os.path.join(folder, '*.png'))
    images = []
    for file in files:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 255.0
        images.append(img)
    return np.array(images)

# -----------------------------
# Model Components
# -----------------------------
def build_encoder(input_shape=(32,32,3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    latent = tf.keras.layers.Dense(128, activation='relu')(x)
    return tf.keras.Model(inputs, latent, name='encoder')

def build_decoder(latent_dim=128):
    inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(8*8*64, activation='relu')(inputs)
    x = tf.keras.layers.Reshape((8,8,64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    return tf.keras.Model(inputs, outputs, name='decoder')

# -----------------------------
# Training setup
# -----------------------------
def train_face_swapper(imagesA, imagesB, epochs=2000, batch_size=32):
    encoder = build_encoder()
    decoder_A = build_decoder()
    decoder_B = build_decoder()

    # Create separate optimizers
    opt_encoder = tf.keras.optimizers.Adam(0.001)
    opt_decoderA = tf.keras.optimizers.Adam(0.001)
    opt_decoderB = tf.keras.optimizers.Adam(0.001)

    loss_fn = tf.keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        idxA = np.random.randint(0, len(imagesA), batch_size)
        idxB = np.random.randint(0, len(imagesB), batch_size)

        batchA = imagesA[idxA]
        batchB = imagesB[idxB]

        with tf.GradientTape(persistent=True) as tape:
            latentA = encoder(batchA)
            latentB = encoder(batchB)

            outputA = decoder_A(latentA)
            outputB = decoder_B(latentB)

            lossA = loss_fn(batchA, outputA)
            lossB = loss_fn(batchB, outputB)

            total_loss = lossA + lossB

        # Compute gradients
        grads_enc = tape.gradient(total_loss, encoder.trainable_variables)
        grads_decA = tape.gradient(lossA, decoder_A.trainable_variables)
        grads_decB = tape.gradient(lossB, decoder_B.trainable_variables)

        # Apply gradients using separate optimizers
        opt_encoder.apply_gradients(zip(grads_enc, encoder.trainable_variables))
        opt_decoderA.apply_gradients(zip(grads_decA, decoder_A.trainable_variables))
        opt_decoderB.apply_gradients(zip(grads_decB, decoder_B.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: LossA={lossA.numpy():.4f}, LossB={lossB.numpy():.4f}")

    # Save models
    os.makedirs("models", exist_ok=True)
    encoder.save_weights("models/simple_encoder.weights.h5")
    decoder_A.save_weights("models/simple_decoder_A.weights.h5")
    decoder_B.save_weights("models/simple_decoder_B.weights.h5")

    print("Training complete. Models saved to /models")

# -----------------------------
# Run training
# -----------------------------
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    imagesA = load_images('dataset/personA')
    imagesB = load_images('dataset/personB')

    # Check for empty dataset
    if len(imagesA) == 0 or len(imagesB) == 0:
        raise ValueError("Dataset folders are empty! Add images to dataset/personA and dataset/personB.")

    train_face_swapper(imagesA, imagesB)
