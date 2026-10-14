import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

def detect_faces(image, face_list):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()

    for face in face_list:
        # Draw bounding box
        x, y, width, height = face['box']
        rect = Rectangle((x, y), width, height, fill=False, color='orange')
        ax.add_patch(rect)

        # Draw landmarks
        for key, value in face['keypoints'].items():
            dot = Circle(value, radius=5, color='red')  # Reduced radius for better visibility
            ax.add_patch(dot)

    plt.axis('off')
    plt.show()

# Instantiate MTCNN detector
detector = MTCNN()

# Load sample image
image = cv2.imread('trump_ref.png')   

# Detect faces and landmarks
faces = detector.detect_faces(image)

# Visualize results
detect_faces(image, faces)
