import face_recognition

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
