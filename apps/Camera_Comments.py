import cv2
import dlib
import numpy as np
import time
from collections import deque

class DrowsinessDetector:
    def __init__(self, yaw_threshold=20.0, pitch_threshold=10.0, ear_threshold=0.25, buffer_size=10):
        """
        Initialize the DrowsinessDetector class with specified thresholds and buffer size for yaw, pitch, and roll.

        Parameters:
        - yaw_threshold: The angle deviation (in degrees) to consider a person looking away (left or right).
        - pitch_threshold: The angle deviation (in degrees) to consider a person looking too far up or down.
        - ear_threshold: The Eye Aspect Ratio (EAR) below which the person is considered drowsy.
        - buffer_size: The number of frames used to smooth out head pose data.
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('predictors/shape_predictor_68_face_landmarks.dat')

        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.ear_threshold = ear_threshold

        self.distracted = False
        self.distracted_start_time = None
        self.total_distracted_time = 0.0

        # Circular queues to store the last N yaw, pitch, and roll values to smooth noisy data.
        self.yaw_buffer = deque(maxlen=buffer_size)
        self.pitch_buffer = deque(maxlen=buffer_size)
        self.roll_buffer = deque(maxlen=buffer_size)

        # Model points for head pose estimation (representing key points on the face).
        self.model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -330.0, -65.0),   # Chin
            (-225.0, 170.0, -135.0), # Left eye corner
            (225.0, 170.0, -135.0),  # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

    def calculate_ear(self, eye):
        """
        Calculate the Eye Aspect Ratio (EAR) to detect drowsiness.

        The EAR is based on the distances between specific points around the eyes. If the ratio falls below
        a certain threshold, it indicates that the eyes are closing (potential drowsiness).

        Parameters:
        - eye: A set of 6 coordinates around the eye landmarks.

        Returns:
        - ear: The Eye Aspect Ratio (EAR) value.
        """
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def rotation_matrix_to_euler_angles(self, R):
        """
        Convert a 3D rotation matrix to Euler angles (pitch, yaw, roll).

        Parameters:
        - R: The 3x3 rotation matrix.

        Returns:
        - pitch, yaw, roll: The Euler angles representing the head's orientation.
        """
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.degrees(x), np.degrees(y), np.degrees(z)  # Convert from radians to degrees

    def adjust_pitch(self, pitch):
        """
        Adjust the pitch value to center around 0 degrees for a forward-looking head.

        This corrects for the pitch angle exceeding 90 degrees, which could happen due to the orientation
        of the head relative to the camera.

        Parameters:
        - pitch: The original pitch value.

        Returns:
        - The adjusted pitch value.
        """
        if pitch > 90:
            pitch = pitch - 180
        return pitch

    def estimate_head_pose(self, landmarks, camera_matrix):
        """
        Estimate the head pose (yaw, pitch, roll) based on facial landmarks.

        Uses the `solvePnP` method to find the rotation vectors and translation vectors that describe the pose.
        Then, converts the rotation vector into Euler angles.

        Parameters:
        - landmarks: The 2D coordinates of facial landmarks from the image.
        - camera_matrix: The camera matrix for solving the pose.

        Returns:
        - pitch, yaw, roll: The head pose angles.
        """
        # 2D image points corresponding to the 3D model points.
        image_points = np.array([
            (landmarks[30][0], landmarks[30][1]),  # Nose tip
            (landmarks[8][0], landmarks[8][1]),    # Chin
            (landmarks[36][0], landmarks[36][1]),  # Left eye corner
            (landmarks[45][0], landmarks[45][1]),  # Right eye corner
            (landmarks[48][0], landmarks[48][1]),  # Left mouth corner
            (landmarks[54][0], landmarks[54][1])   # Right mouth corner
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))  # No lens distortion
        success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs)

        # Convert the rotation vector to a rotation matrix.
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = self.rotation_matrix_to_euler_angles(rotation_matrix)

        # Adjust the pitch to correct for camera orientation.
        pitch = self.adjust_pitch(pitch)

        return pitch, yaw, roll

    def update_buffer(self, pitch, yaw, roll):
        """
        Update the circular buffers with the new pitch, yaw, and roll values.
        This helps smooth the data by averaging out small variations over time.

        Parameters:
        - pitch, yaw, roll: The head pose angles to store.
        """
        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)
        self.roll_buffer.append(roll)

    def get_averaged_pose(self):
        """
        Return the averaged yaw, pitch, and roll values from the buffer.

        Returns:
        - avg_pitch, avg_yaw, avg_roll: The smoothed values of pitch, yaw, and roll.
        """
        avg_pitch = sum(self.pitch_buffer) / len(self.pitch_buffer) if len(self.pitch_buffer) > 0 else 0
        avg_yaw = sum(self.yaw_buffer) / len(self.yaw_buffer) if len(self.yaw_buffer) > 0 else 0
        avg_roll = sum(self.roll_buffer) / len(self.roll_buffer) if len(self.roll_buffer) > 0 else 0
        return avg_pitch, avg_yaw, avg_roll

    def check_distraction(self, yaw, pitch):
        """
        Check if the user is distracted based on yaw and pitch exceeding the thresholds.

        Parameters:
        - yaw, pitch: The head pose angles to compare against the thresholds.

        Returns:
        - True if the user is distracted, False otherwise.
        """
        return not (abs(yaw) <= self.yaw_threshold and abs(pitch) <= self.pitch_threshold)

    def process_frame(self, frame, camera_matrix):
        """
        Process a single frame, detecting the user's face, eyes, head pose, and checking for distraction and drowsiness.
        Displays the yaw, pitch, roll, and distraction/drowsiness status on the frame.

        Parameters:
        - frame: The current frame from the video feed.
        - camera_matrix: The camera matrix for pose estimation.

        Returns:
        - The processed frame with annotations.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            # No face detected: assume distracted.
            self.distracted = True
            cv2.putText(frame, "Face not visible: Distracted", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        for face in faces:
            landmarks = self.predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            if not self.are_essential_landmarks_visible(landmarks):
                # If essential landmarks are missing, assume distraction.
                self.distracted = True
                cv2.putText(frame, "Partial face: Distracted", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue

            # Estimate head pose based on landmarks.
            pitch, yaw, roll = self.estimate_head_pose(landmarks, camera_matrix)
            self.update_buffer(pitch, yaw, roll)
            avg_pitch, avg_yaw, avg_roll = self.get_averaged_pose()

            # EAR calculation for both eyes to detect drowsiness.
            left_eye = landmarks[36:42]
            right_eye = landmarks[42:48]
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < self.ear_threshold:
                cv2.putText(frame, "Drowsy", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Awake", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check distraction based on yaw and pitch.
            self.distracted = self.check_distraction(avg_yaw, avg_pitch)

            if self.distracted:
                if self.distracted_start_time is None:
                    self.distracted_start_time = time.time()
                distracted_time = time.time() - self.distracted_start_time
                self.total_distracted_time += distracted_time
                cv2.putText(frame, f"Distracted Time: {distracted_time:.2f} sec", (50, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                self.distracted_start_time = None

            # Display yaw, pitch, and roll on the frame.
            cv2.putText(frame, f"Yaw: {avg_yaw:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {avg_pitch:.2f}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Roll: {avg_roll:.2f}", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display distraction status and total distracted time.
            cv2.putText(frame, f"Distracted: {self.distracted}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if self.distracted else (255, 255, 255), 2)
            cv2.putText(frame, f"Total Distracted Time: {self.total_distracted_time:.2f} sec", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw the head direction line to show where the head is pointing.
            dist_coeffs = np.zeros((4, 1))  # No lens distortion
            nose_end_point3D = np.array([[0, 0, 1000.0]], dtype='float32')  # Point in the direction of the nose.
            nose_end_point2D, _ = cv2.projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            p1 = (int(landmarks[30][0]), int(landmarks[30][1]))  # Nose tip
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))  # Projected nose tip
            cv2.line(frame, p1, p2, (255, 0, 0), 2)  # Draw a blue line showing head direction.

        return frame

    def are_essential_landmarks_visible(self, landmarks):
        """
        Check if essential landmarks (nose, eyes, chin) are visible.
        If any of these landmarks are out of frame, assume the user is distracted.

        Parameters:
        - landmarks: The facial landmarks detected.

        Returns:
        - True if essential landmarks are visible, False otherwise.
        """
        for i in [30, 8, 36, 45]:  # Nose tip, chin, left eye, right eye.
            if landmarks[i][0] <= 0 or landmarks[i][1] <= 0:
                return False
        return True

    def run(self):
        """
        Main loop to capture video feed, process each frame, and display the processed output.
        """
        cap = cv2.VideoCapture(0)
        focal_length = cap.get(3)
        center = (cap.get(3) / 2, cap.get(4) / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process the current frame to detect drowsiness and distraction.
            processed_frame = self.process_frame(frame, camera_matrix)
            cv2.imshow("Drowsiness Detector", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()
