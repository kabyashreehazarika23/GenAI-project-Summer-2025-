import cv2
import numpy as np
import face_recognition
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
import random
import os

# 1️⃣ Face Detection & Extraction

class DetectedFace:
    def __init__(self, image, box, landmarks):
        self.image = image
        self.box = box
        self.landmarks = landmarks

# Load DLIB Landmark model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detect_faces(image):
    locations = face_recognition.face_locations(image)
    faces = []

    for (top, right, bottom, left) in locations:
        face_crop = image[top:bottom, left:right]
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = predictor(image, rect)
        faces.append(DetectedFace(face_crop, (top, right, bottom, left), landmarks))
    return faces

# 2️⃣ Face Recognition Filter

class FaceFilter:
    def __init__(self, reference_image_path, threshold=0.6):
        ref_img = face_recognition.load_image_file(reference_image_path)
        self.ref_encoding = face_recognition.face_encodings(ref_img)[0]
        self.threshold = threshold

    def check(self, face):
        encodings = face_recognition.face_encodings(face.image)
        if len(encodings) > 0:
            distance = face_recognition.face_distance([self.ref_encoding], encodings[0])[0]
            return distance < self.threshold
        return False

# 3️⃣ Data Augmentation Function

def augment_image(image):
    h, w = image.shape[:2]
    center = (w//2, h//2)

    # Random rotation and scale
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.9, 1.1)

    # Affine transformation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    augmented = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Random flip
    if random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)

    return augmented

# 4️⃣ Main Execution

if __name__ == "__main__":
    # Input images
    image = cv2.imread(r" group_pht.png")   # Image with multiple faces
    reference = 'trump_ref.png'              # Reference face of Trump

    if image is None:
        print("Error: group_photo.png not found.")
        exit()

    # Detect faces
    faces = detect_faces(image)

    # Apply FaceFilter
    filter = FaceFilter(reference)

    for i, face in enumerate(faces):
        plt.figure(figsize=(10,3))

        plt.subplot(1,3,1)
        plt.title("Detected Face")
        plt.imshow(cv2.cvtColor(face.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        if filter.check(face):
            plt.subplot(1,3,2)
            plt.title("Matched with Trump")
            plt.imshow(cv2.cvtColor(face.image, cv2.COLOR_BGR2RGB))
            plt.axis('off')

            # Augment and show
            aug = augment_image(face.image)
            plt.subplot(1,3,3)
            plt.title("Augmented Face")
            plt.imshow(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
            plt.axis('off')

        plt.show()
