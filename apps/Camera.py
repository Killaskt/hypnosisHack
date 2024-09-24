import cv2
import dlib
import numpy as np
import time
from collections import deque

# possibly add an EAR optimization for better drowsiness detection methods

class DrowsinessDetector:
    def __init__(self, yaw_threshold=20.0, pitch_threshold=10.0, ear_threshold=0.25, buffer_size=10, frame_skip=2,
                 hypo_low_duration=30, hypo_medium_duration=35, hypo_high_duration=40, movement_tolerance=7,
                 drowsiness_confidence_threshold=2.0, distraction_time_threshold=3):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('predictors/shape_predictor_68_face_landmarks.dat')

        # Thresholds
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.ear_threshold = ear_threshold
        self.frame_skip = frame_skip  # Skip frames for performance optimization

        # Time thresholds for hypnosis states (low, mid, high)
        self.hypo_low_duration = hypo_low_duration  # 30 seconds
        self.hypo_medium_duration = hypo_medium_duration  # 35 seconds
        self.hypo_high_duration = hypo_high_duration  # 40 seconds

        # Drowsiness confidence and distraction time threshold
        self.drowsiness_confidence_threshold = drowsiness_confidence_threshold  # 60% confidence
        self.distraction_time_threshold = distraction_time_threshold  # 15 seconds

        self.movement_tolerance = movement_tolerance  # Tolerance for small movements in yaw/pitch

        # Distraction variables
        self.distracted = False
        self.distracted_start_time = None
        self.total_distracted_time = 0.0

        # Hypnosis variables
        self.hypnotized = None  # None, 'low', 'medium', or 'high'
        self.fixation_start_time = None

        # Drowsiness confidence variables
        self.drowsy_confidence = 0.0
        self.drowsy_time = None  # Start time for detecting drowsiness
        self.drowsy_bool = False  # Final bool for drowsiness

        # Circular buffers to smooth yaw, pitch, and roll
        self.yaw_buffer = deque(maxlen=buffer_size)
        self.pitch_buffer = deque(maxlen=buffer_size)
        self.roll_buffer = deque(maxlen=buffer_size)

        # Model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -330.0, -65.0),   # Chin
            (-225.0, 170.0, -135.0), # Left eye corner
            (225.0, 170.0, -135.0),  # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Buffers for fixation detection (for hypnosis detection)
        self.last_gaze_positions = deque(maxlen=30)  # Buffer for recent yaw/pitch
        self.blink_times = deque(maxlen=10)  # Buffer for blink timestamps

        # Frame counter to skip frames
        self.frame_count = 0

        # Variables to store processed values between skipped frames
        self.avg_pitch = 0
        self.avg_yaw = 0
        self.avg_roll = 0
        self.ear = 0
        self.rotation_vector = None
        self.translation_vector = None
        self.landmarks = None  # To store landmarks between frames

    def calculate_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def rotation_matrix_to_euler_angles(self, R):
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

        return np.degrees(x), np.degrees(y), np.degrees(z)

    def adjust_pitch(self, pitch):
        # Adjust the pitch value to bring it to a neutral position
        if pitch > 90:
            pitch = pitch - 180
        elif pitch < -90:
            pitch = pitch + 180
        return pitch

    def normalize_angle(self, angle):
        # Normalize an angle to the range -180 to 180 degrees
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def estimate_head_pose(self, landmarks, camera_matrix):
        # Estimate head pose using landmarks and solvePnP
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

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = self.rotation_matrix_to_euler_angles(rotation_matrix)

        pitch = self.adjust_pitch(pitch)
        yaw = self.normalize_angle(yaw)

        return pitch, yaw, roll, rotation_vector, translation_vector

    def update_buffer(self, pitch, yaw, roll):
        # Add pitch, yaw, and roll values to circular buffers
        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)
        self.roll_buffer.append(roll)

    def get_averaged_pose(self):
        # Return average yaw, pitch, and roll values from buffers to smooth noise
        avg_pitch = sum(self.pitch_buffer) / len(self.pitch_buffer) if len(self.pitch_buffer) > 0 else 0
        avg_yaw = sum(self.yaw_buffer) / len(self.yaw_buffer) if len(self.yaw_buffer) > 0 else 0
        avg_roll = sum(self.roll_buffer) / len(self.roll_buffer) if len(self.roll_buffer) > 0 else 0
        return avg_pitch, avg_yaw, avg_roll

    def check_distraction(self, yaw, pitch):
        # Check if the user is distracted based on yaw and pitch thresholds
        return not (abs(yaw) <= self.yaw_threshold and abs(pitch) <= self.pitch_threshold)

    def update_hypnotized_state(self, fixation_duration):
        # Update the hypnosis state based on the duration of fixation
        if fixation_duration >= self.hypo_high_duration:
            self.hypnotized = "high"
        elif fixation_duration >= self.hypo_medium_duration:
            self.hypnotized = "medium"
        elif fixation_duration >= self.hypo_low_duration:
            self.hypnotized = "low"
        else:
            self.hypnotized = None

    def process_frame(self, frame, camera_matrix):
        # Process each frame for head pose, distraction, drowsiness, and hypnosis detection
        self.frame_count += 1

        if self.frame_count % self.frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if len(faces) == 0:
                self.distracted = True
                cv2.putText(frame, "Face not visible: Distracted", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame

            for face in faces:
                self.landmarks = self.predictor(gray, face)
                self.landmarks = np.array([[p.x, p.y] for p in self.landmarks.parts()])

                if not self.are_essential_landmarks_visible(self.landmarks):
                    self.distracted = True
                    cv2.putText(frame, "Partial face: Distracted", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    continue

                # Estimate head pose if landmarks are visible
                pitch, yaw, roll, rotation_vector, translation_vector = self.estimate_head_pose(self.landmarks, camera_matrix)
                self.update_buffer(pitch, yaw, roll)

                self.avg_pitch, self.avg_yaw, self.avg_roll = self.get_averaged_pose()

                left_eye = self.landmarks[36:42]
                right_eye = self.landmarks[42:48]
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                self.ear = (left_ear + right_ear) / 2.0

                # Drowsiness detection with confidence logic
                if self.ear < self.ear_threshold:
                    if self.drowsy_time is None:
                        self.drowsy_time = time.time()

                    self.drowsy_confidence = (time.time() - self.drowsy_time) / 3  # Example confidence logic

                    if self.drowsy_confidence >= self.drowsiness_confidence_threshold:
                        self.drowsy_bool = True
                    else:
                        self.drowsy_bool = False
                else:
                    self.drowsy_time = None
                    self.drowsy_confidence = 0
                    self.drowsy_bool = False

                # Distraction detection with time threshold
                self.distracted = self.check_distraction(self.avg_yaw, self.avg_pitch)
                if self.distracted:
                    if self.distracted_start_time is None:
                        self.distracted_start_time = time.time()
                    distracted_duration = time.time() - self.distracted_start_time
                    if distracted_duration >= self.distraction_time_threshold:
                        self.distracted = True
                    else:
                        self.distracted = False
                else:
                    self.distracted_start_time = None

                # Hypnosis detection logic
                self.last_gaze_positions.append((self.avg_yaw, self.avg_pitch))

                if all(abs(self.avg_yaw - pos[0]) < self.movement_tolerance and abs(self.avg_pitch - pos[1]) < self.movement_tolerance for pos in self.last_gaze_positions):
                    if self.fixation_start_time is None:
                        self.fixation_start_time = time.time()

                    fixation_duration = time.time() - self.fixation_start_time
                    self.update_hypnotized_state(fixation_duration)

                else:
                    self.fixation_start_time = None
                    self.hypnotized = None

        # Display updates, including hypnosis state and fixation timer
        fixation_duration_display = 0 if self.fixation_start_time is None else time.time() - self.fixation_start_time
        cv2.putText(frame, f"Fixation Time: {fixation_duration_display:.2f} sec", (950, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Hypnotized: {self.hypnotized if self.hypnotized else 'None'}", (950, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if self.hypnotized else (255, 255, 255), 2)

        # Display yaw, pitch, roll, and EAR logs on screen
        cv2.putText(frame, f"Yaw: {self.avg_yaw:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Pitch: {self.avg_pitch:.2f}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Roll: {self.avg_roll:.2f}", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"EAR: {self.ear:.2f}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display distraction state and total distraction time
        distracted_time = 0 if self.distracted_start_time is None else time.time() - self.distracted_start_time
        cv2.putText(frame, f"Distracted: {self.distracted}", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if self.distracted else (255, 255, 255), 2)
        cv2.putText(frame, f"Distracted Time: {distracted_time:.2f} sec", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Distracted Time: {self.total_distracted_time:.2f} sec", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display drowsiness confidence
        cv2.putText(frame, f"Drowsy: {self.drowsy_bool}", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if self.drowsy_bool else (255, 255, 255), 2)
        cv2.putText(frame, f"Drowsy Confidence: {self.drowsy_confidence:.2f}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def are_essential_landmarks_visible(self, landmarks):
        # Check if key landmarks like the nose, chin, and eyes are visible
        for i in [30, 8, 36, 45]:
            if landmarks[i][0] <= 0 or landmarks[i][1] <= 0:
                return False
        return True

    def run(self):
        # Main loop to capture frames and run the detection logic
        cap = cv2.VideoCapture(1)
        focal_length = cap.get(3)
        center = (cap.get(3) / 2, cap.get(4) / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame, camera_matrix)
            cv2.imshow("Drowsiness Detector", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()
