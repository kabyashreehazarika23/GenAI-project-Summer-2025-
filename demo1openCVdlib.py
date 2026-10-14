import cv2
import dlib

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
image = cv2.imread(r"C:\Users\91863\OneDrive\Desktop\PLS Code\nicole.png")


# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(gray)

# Loop through each detected face
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    # Predict facial landmarks
    landmarks = predictor(gray, face)

    # Draw circles on each landmark point
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Show the image with landmarks
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

