 
import cv2
import dlib
import glob
import my_face_recognition
import face_recognition_models
from face_utils import Extract 
from face_utils import FaceFilter 
from face_utils import DetectedFace
from matplotlib import pyplot as plt

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor = dlib.shape_predictor(predictor_68_point_model)

def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])
    
def _raw_face_landmarks(face_image, face_locations):
    face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def detect_faces(frame):
    face_locations = my_face_recognition.face_locations(frame)
    landmarks = _raw_face_landmarks(frame, face_locations)

    for ((y, right, bottom, x), landmarks) in zip(face_locations, landmarks):
        yield DetectedFace(frame[y: bottom, x: right], 
                           x, right - x, y, bottom - y, landmarks)

# Load Sample Image
image = cv2.imread(r"C:\Users\91863\OneDrive\Desktop\PLS Code\sample_image_cc (1).png")
plt.imshow(cv2.cvtColor(image , cv2.COLOR_BGR2RGB))
plt.axis('off');
# Detect faces and visualize them all
detected_faces = [face for face in detect_faces(image)]
for face in detected_faces:
  plt.imshow(cv2.cvtColor(face.image , cv2.COLOR_BGR2RGB))
  plt.axis('off');
  plt.show()

# Reference Image
ref_image = cv2.imread('trump_ref_cc.jpg')
plt.imshow(cv2.cvtColor(ref_image , cv2.COLOR_BGR2RGB))
plt.axis('off');
     

face_filter = FaceFilter('trump_ref_cc.jpg')

for face in detected_faces:
  if face_filter.check(face):
    plt.title("Matching Face")
    plt.imshow(cv2.cvtColor(face.image , cv2.COLOR_BGR2RGB))
    plt.axis('off');
    plt.show() 
extractor = Extract()
resized_image = extractor.extract(image, detected_faces[0], 256)
plt.imshow(cv2.cvtColor(resized_image , cv2.COLOR_BGR2RGB))
plt.axis('off');
plt.show()   