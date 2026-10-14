import os
import cv2
import numpy as np
import tensorflow as tf

# -----------------------------
# Model Definitions (Same as Training)
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
# Face Swap Function
# -----------------------------
def face_swap(input_image_path, output_image_path):
    # Load models
    encoder = build_encoder()
    decoder_B = build_decoder()  # Use decoder_B to swap person A -> person B

    encoder.load_weights("models/simple_encoder.weights.h5")
    decoder_B.load_weights("models/simple_decoder_B.weights.h5")


    # Load input image
    img = cv2.imread(input_image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (32,32))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    # Encode
    latent = encoder.predict(img_batch)

    # Decode using decoder_B (swap face)
    swapped = decoder_B.predict(latent)[0]

    # Convert back to image
    swapped_img = (swapped * 255).astype(np.uint8)
    swapped_img = cv2.cvtColor(swapped_img, cv2.COLOR_RGB2BGR)

    # Save result
    cv2.imwrite(output_image_path, swapped_img)
    print(f"Swapped face saved to {output_image_path}")

# -----------------------------
# Run the swap
# -----------------------------
if __name__ == "__main__":
    input_image = "dataset/personA/test_image.jpg"  # Input: Person A's image
    output_image = "outputs/swapped_to_B.jpg"        # Output: Swapped to Person B

    os.makedirs('outputs', exist_ok=True)
    face_swap(input_image, output_image)
