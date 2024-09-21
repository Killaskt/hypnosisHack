import cv2
import dlib
import numpy as np

#
# Checkpoint: drowsiness and posture works
#

# Load the pre-trained face detector and facial landmarks predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download this from dlib
# predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')  # Download this from dlib

# Camera setup
cap = cv2.VideoCapture(1)

# 3D model points (reference facial points for head pose estimation)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0), # Left eye corner
    (225.0, 170.0, -135.0),  # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

# Camera matrix (approximation, should be calibrated for more accuracy)
focal_length = cap.get(3)  # Assume focal length is width of video
center = (cap.get(3) / 2, cap.get(4) / 2)
camera_matrix = np.array([[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype="double")

# Define EAR function for eye detection
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Loop to process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        # Get the facial landmarks
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Select the landmarks for head pose estimation (nose, chin, eyes, and mouth corners)
        image_points = np.array([
            (landmarks[30][0], landmarks[30][1]),  # Nose tip
            (landmarks[8][0], landmarks[8][1]),    # Chin
            (landmarks[36][0], landmarks[36][1]),  # Left eye corner
            (landmarks[45][0], landmarks[45][1]),  # Right eye corner
            (landmarks[48][0], landmarks[48][1]),  # Left mouth corner
            (landmarks[54][0], landmarks[54][1])   # Right mouth corner
        ], dtype="double")

        # Estimate the head pose using solvePnP
        dist_coeffs = np.zeros((4, 1))  # No lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

        # Project a 3D point (like the nose) back to 2D to visualize the direction of the face
        nose_end_point3D = np.array([[0, 0, 1000.0]], dtype='float32')
        nose_end_point2D, _ = cv2.projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # Draw a line showing head direction
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # Detect eyes for EAR (for simplicity, this uses fixed points for left and right eyes)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # Compute EAR for both eyes
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        ear = (left_ear + right_ear) / 2.0

        # Define a threshold for drowsiness (typically 0.2-0.25)
        if ear < 0.25:
            cv2.putText(frame, "Drowsy", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Awake", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
