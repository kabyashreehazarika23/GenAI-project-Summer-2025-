import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained generator
model_path = r'models/saved_model/generator_epoch1.h5'  # Use raw string for Windows path
generator = load_model(model_path)

# Define folders
landmarks_folder = 'dataset/landmarks'
output_folder = 'outputs/reenacted_faces'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop over landmarks
for img_name in os.listdir(landmarks_folder):
    landmark_path = os.path.join(landmarks_folder, img_name)

    # Load and preprocess landmark image
    landmark_img = cv2.imread(landmark_path)
    if landmark_img is None:
        print(f"Skipping {img_name}, unable to load image.")
        continue

    landmark_img = cv2.resize(landmark_img, (64, 64))
    landmark_img = landmark_img / 127.5 - 1.0  # Normalize to [-1, 1]
    landmark_img = np.expand_dims(landmark_img, axis=0)  # Add batch dimension

    # Generate reenacted face
    generated_img = generator.predict(landmark_img)
    generated_img = (generated_img[0] + 1) * 127.5  # Denormalize to [0, 255]
    generated_img = np.clip(generated_img, 0, 255).astype(np.uint8)

    # Save reenacted face
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, cv2.cvtColor(generated_img, cv2.COLOR_RGB2BGR))

print(f"Reenactment completed! Check the folder: {output_folder}")
