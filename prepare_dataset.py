import cv2
import dlib
import glob
import os
import face_recognition
import face_recognition_models
from face_utils import Extract, FaceFilter, DetectedFace

# Load DLIB predictor
predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor = dlib.shape_predictor(predictor_68_point_model)

def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def _raw_face_landmarks(face_image, face_locations):
    face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def detect_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    landmarks = _raw_face_landmarks(frame, face_locations)
    for ((top, right, bottom, left), landmarks) in zip(face_locations, landmarks):
        face_crop = frame[top:bottom, left:right]
        w, h = right - left, bottom - top
        yield DetectedFace(face_crop, left, w, top, h, landmarks)

def get_faces(reference_image, image, extractor):
    faces_count = 0
    facefilter = FaceFilter(reference_image)
    for face in detect_faces(image):
        if not facefilter.check(face):
            continue
        yield faces_count, face
        faces_count += 1

def create_face_dataset(reference_face_filepath, input_dir, output_dir, extractor, extensions=['jpg', 'png', 'jpeg']):
    os.makedirs(output_dir, exist_ok=True)
    image_list = [fn for fn in glob.glob(input_dir+"/*.*") if any(fn.lower().endswith(ext) for ext in extensions)]
    print("Total Images to Scan={}".format(len(image_list)))
    positive_ctr = 0

    for filename in image_list:
        try:
            image = cv2.imread(filename)
            for idx, face in get_faces(reference_face_filepath, image, extractor):
                resized_image = extractor.extract(image, face, 256)
                output_file = os.path.join(output_dir, os.path.basename(filename))
                cv2.imwrite(output_file, resized_image)
                positive_ctr += 1
        except Exception as e:
            print('Failed to extract from image: {}. Reason: {}'.format(filename, e))
    print("Images with reference face={}".format(positive_ctr))

if __name__ == "__main__":
    extractor = Extract()
    create_face_dataset("nicolas_ref.png", "nicolas", "nicolas_face", extractor)
    create_face_dataset("trump_ref.png", "trump", "trump_face", extractor)
