# deepfake_dataset_pipeline.py

import cv2
import dlib
import face_recognition
import numpy as np
import glob
import os
from imutils import face_utils

# 1️ DetectedFace Class
class DetectedFace:
    def __init__(self, image, x, w, y, h, landmarks):
        self.image = image
        self.x = x
        self.w = w
        self.y = y
        self.h = h
        self.landmarks = landmarks

    def landmarksAsXY(self):
        return [(p.x, p.y) for p in self.landmarks.parts()]

# 2️ FaceFilter Class
class FaceFilter:
    def __init__(self, reference_file_path, threshold=0.65):
        image = face_recognition.load_image_file(reference_file_path)
        self.encoding = face_recognition.face_encodings(image)[0]
        self.threshold = threshold

    def check(self, detected_face):
        encodings = face_recognition.face_encodings(detected_face.image)
        if len(encodings) > 0:
            score = face_recognition.face_distance([self.encoding], encodings[0])
        else:
            return False
        return score <= self.threshold

# 3️ Data Augmentation

def random_transform(image, rotation_range=5, zoom_range=0.05, shift_range=0.05, random_flip=0.5):
    h, w = image.shape[:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    M[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result

# 4️ Detect Faces

predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

def detect_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    landmarks_list = []
    for (top, right, bottom, left) in face_locations:
        rect = dlib.rectangle(left, top, right, bottom)
        landmarks = predictor(frame, rect)
        landmarks_list.append(DetectedFace(frame[top:bottom, left:right],
                                           left, right-left, top, bottom-top, landmarks))
    return landmarks_list

# 5️ Extract Faces and Create Dataset

def extract_faces(reference_image_path, input_dir, output_dir):
    face_filter = FaceFilter(reference_image_path)

    image_files = glob.glob(os.path.join(input_dir, "*.*"))
    for img_file in image_files:
        image = cv2.imread(img_file)
        detected_faces = detect_faces(image)

        for face in detected_faces:
            if face_filter.check(face):
                resized = cv2.resize(face.image, (256, 256))
                save_path = os.path.join(output_dir, os.path.basename(img_file))
                cv2.imwrite(save_path, resized)

                # Optional augmentation
                aug = random_transform(resized)
                aug_path = os.path.join(output_dir, 'aug_' + os.path.basename(img_file))
                cv2.imwrite(aug_path, aug)

if __name__ == "__main__":
    reference_image = r"C:\Users\91863\OneDrive\Desktop\PLS Code\trump_ref.png"
    input_folder = "input_images"
    output_folder = "output_faces"
    os.makedirs(output_folder, exist_ok=True)
    extract_faces(reference_image, input_folder, output_folder)
    print("Dataset creation completed.")
import matplotlib.pyplot as plt

for img_file in os.listdir(output_folder):
    img = cv2.imread(os.path.join(output_folder, img_file))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
