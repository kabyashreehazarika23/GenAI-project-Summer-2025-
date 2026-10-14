import cv2
import numpy as np
import face_recognition
from imutils import face_utils

class DetectedFace:
    def __init__(self, image, x, w, y, h, landmarks):
        self.image = image
        self.box = (y, x + w, y + h, x)  # (top, right, bottom, left)
        self.w = w
        self.h = h
        self.landmarks = landmarks

class FaceFilter:
    def __init__(self, reference_file_path, threshold=0.65):
        image = face_recognition.load_image_file(reference_file_path)
        self.encoding = face_recognition.face_encodings(image)[0]
        self.threshold = threshold

    def check(self, detected_face):
        # Use full crop with (0, width, height, 0)
        face_location = [(0, detected_face.w, detected_face.h, 0)]
        encodings = face_recognition.face_encodings(detected_face.image, face_location)

        if len(encodings) > 0:
            distance = face_recognition.face_distance([self.encoding], encodings[0])[0]
            return distance < self.threshold
        else:
            return False

class Extract:
    def extract(self, image, face, size=256):
        points = face_utils.shape_to_np(face.landmarks)
        eyes_center = np.mean(points[36:48], axis=0).astype(int)
        M = cv2.getRotationMatrix2D(tuple(eyes_center), 0, 1.0)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        face_crop = aligned[face.box[0]:face.box[2], face.box[3]:face.box[1]]
        return cv2.resize(face_crop, (size, size))
