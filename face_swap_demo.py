import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import sys
import os

# Load Models
encoder = load_model('models/encoder.h5', compile=False)
decoder_trump = load_model('models/decoder_trump.h5', compile=False)
decoder_cage = load_model('models/decoder_cage.h5', compile=False)

def swap_face(img, target_decoder):
    if img is None:
        raise FileNotFoundError("Input image not found or cannot be read. Please check the file path.")
    
    img_input = cv2.resize(img, (64, 64)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    embedding = encoder.predict(img_input)
    output = target_decoder.predict(embedding)
    
    output_img = np.clip(output[0] * 255.0, 0, 255).astype(np.uint8)
    
    return cv2.resize(output_img, (img.shape[1], img.shape[0]))

# Main Execution
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python face_swap_demo.py <input_image> <output_image> <swap_direction>")
        print("swap_direction: trump2cage or cage2trump")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    direction = sys.argv[3]

    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)

    img = cv2.imread(input_path)

    if direction == 'trump2cage':
        result = swap_face(img, decoder_cage)
    elif direction == 'cage2trump':
        result = swap_face(img, decoder_trump)
    else:
        print("Invalid swap_direction! Use 'trump2cage' or 'cage2trump'")
        sys.exit(1)

    cv2.imwrite(output_path, result)
    print(f"Face swapped image saved to {output_path}")
