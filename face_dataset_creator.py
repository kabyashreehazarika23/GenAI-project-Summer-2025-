import face_recognition
import cv2
import os
import numpy as np

# Paths
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--person', type=str, required=True, help="Person name: trump or cage")
args = parser.parse_args()

ref_image_path = f'{args.person}_ref.png'
input_folder = f'raw_images/{args.person}'
output_folder = f'dataset/{args.person}_faces'


# Create output directory if not exists
os.makedirs(output_folder, exist_ok=True)

# Load reference image and encode it
ref_image = face_recognition.load_image_file(ref_image_path)
ref_encoding = face_recognition.face_encodings(ref_image)[0]

# Threshold for face similarity (lower is stricter)
THRESHOLD = 0.5

# Process all images in the folder
images = [img for img in os.listdir(input_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Total Images to Scan = {len(images)}")

saved_count = 0

for img_name in images:
    img_path = os.path.join(input_folder, img_name)
    image = face_recognition.load_image_file(img_path)

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if len(face_locations) == 0:
        print(f"No faces found in {img_name}. Skipping.")
        continue

    for location, encoding in zip(face_locations, face_encodings):
        # Compare with reference encoding
        distance = np.linalg.norm(ref_encoding - encoding)

        if distance < THRESHOLD:
            top, right, bottom, left = location
            face_crop = image[top:bottom, left:right]

            # Save the cropped face
            output_path = os.path.join(output_folder, f"{saved_count}_{img_name}")
            cv2.imwrite(output_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
            saved_count += 1

print(f"Images with matching face saved: {saved_count}")
