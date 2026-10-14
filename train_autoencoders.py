import numpy as np
import cv2
import os
from utils.face_utils import load_images, build_autoencoder, save_model, train

# Load datasets
cage_images = load_images("dataset/cage/")
trump_images = load_images("dataset/trump/")

# Build autoencoder (simple fully connected for demo purpose)
encoder, decoder_A = build_autoencoder()
_, decoder_B = build_autoencoder()

# Train for Nicolas Cage (Autoencoder A)
train(encoder, decoder_A, cage_images, "Cage Autoencoder")

# Train decoder B for Donald Trump faces (with shared encoder)
train(encoder, decoder_B, trump_images, "Trump Decoder")

# Save models
save_model(encoder, "models/encoder_A.npy")
save_model(decoder_A, "models/decoder_A.npy")
save_model(decoder_B, "models/decoder_B.npy")
