import dlib
import cv2
import face_recognition
from imutils import face_utils

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

# Load model
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
