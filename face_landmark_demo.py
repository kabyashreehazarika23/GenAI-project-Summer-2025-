import cv2
import dlib

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()


predictor = dlib.shape_predictor(r"C:\Users\91863\OneDrive\Desktop\PLS Code\shape_predictor_68_face_landmarks.dat")

# Load the image 
image = cv2.imread("nicole.png")

# corrected function name and color conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Loop through each face found in the image
for face in faces:
    # Get facial landmarks
    landmarks = predictor(gray, face)
    
    # Draw landmarks on the image
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Display the output image
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()