import cv2
import dlib
import numpy as np
import time
from collections import deque
from queue import Queue

class Camera:
    def __init__(self, output_queue, yaw_threshold=20.0, pitch_threshold=10.0, ear_threshold=0.25, buffer_size=10, frame_skip=2,
                 hypo_low_duration=5, hypo_medium_duration=10, hypo_high_duration=15, movement_tolerance=7,
                 drowsiness_confidence_threshold=.5, distraction_time_threshold=1):
        # Initialize dlib face detector and predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('predictors/shape_predictor_68_face_landmarks.dat')

        # Assign the output queue
        self.output_queue = output_queue

        # Thresholds
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.ear_threshold = ear_threshold
        self.frame_skip = frame_skip  # Skip frames for performance optimization

        # Time thresholds for hypnosis states (low, mid, high)
        self.hypo_low_duration = hypo_low_duration
        self.hypo_medium_duration = hypo_medium_duration
        self.hypo_high_duration = hypo_high_duration

        # Drowsiness confidence and distraction time threshold
        self.drowsiness_confidence_threshold = drowsiness_confidence_threshold
        self.distraction_time_threshold = distraction_time_threshold

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

        # Initialize model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -330.0, -65.0),   # Chin
            (-225.0, 170.0, -135.0), # Left eye corner
            (225.0, 170.0, -135.0),  # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Buffers for fixation detection (for hypnosis detection)
        self.last_gaze_positions = deque(maxlen=30)
        self.blink_times = deque(maxlen=10)

        # Frame counter to skip frames
        self.frame_count = 0

        # Variables to store processed values between skipped frames
        self.avg_pitch = 0.0
        self.avg_yaw = 0.0
        self.avg_roll = 0.0
        self.ear = 0.0
        self.rotation_vector = None
        self.translation_vector = None
        self.landmarks = None  # To store landmarks between frames

    def calculate_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

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
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
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
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs)
            if not success:
                return None, None, None
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, roll = self.rotation_matrix_to_euler_angles(rotation_matrix)

            pitch = self.adjust_pitch(pitch)
            yaw = self.normalize_angle(yaw)
            return pitch, yaw, roll
        except cv2.error as e:
            print(f"OpenCV error in solvePnP: {e}")
            return None, None, None

    def update_buffer(self, pitch, yaw, roll):
        self.pitch_buffer.append(float(pitch))
        self.yaw_buffer.append(float(yaw))
        self.roll_buffer.append(float(roll))

    def get_averaged_pose(self):
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
                return

            for face in faces:
                self.landmarks = self.predictor(gray, face)
                self.landmarks = np.array([[p.x, p.y] for p in self.landmarks.parts()])

                pitch, yaw, roll = self.estimate_head_pose(self.landmarks, camera_matrix)
                if pitch is not None and yaw is not None:
                    self.update_buffer(pitch, yaw, roll)

                    self.avg_pitch, self.avg_yaw, self.avg_roll = self.get_averaged_pose()

                    left_eye = self.landmarks[36:42]
                    right_eye = self.landmarks[42:48]
                    left_ear = self.calculate_ear(left_eye)
                    right_ear = self.calculate_ear(right_eye)
                    self.ear = (left_ear + right_ear) / 2.0

                    # Drowsiness detection
                    if self.ear < self.ear_threshold:
                        if self.drowsy_time is None:
                            self.drowsy_time = time.time()

                        self.drowsy_confidence = (time.time() - self.drowsy_time) / 3

                        self.drowsy_bool = self.drowsy_confidence >= self.drowsiness_confidence_threshold
                    else:
                        self.drowsy_time = None
                        self.drowsy_confidence = 0
                        self.drowsy_bool = False

                    # Distraction detection
                    self.distracted = self.check_distraction(self.avg_yaw, self.avg_pitch)
                    if self.distracted:
                        if self.distracted_start_time is None:
                            self.distracted_start_time = time.time()
                        distracted_duration = time.time() - self.distracted_start_time
                        self.distracted = distracted_duration >= self.distraction_time_threshold
                    else:
                        self.distracted_start_time = None

                    # Hypnosis detection
                    self.last_gaze_positions.append((self.avg_yaw, self.avg_pitch))
                    if all(abs(self.avg_yaw - pos[0]) < self.movement_tolerance for pos in self.last_gaze_positions):
                        if self.fixation_start_time is None:
                            self.fixation_start_time = time.time()
                        fixation_duration = time.time() - self.fixation_start_time
                        self.update_hypnotized_state(fixation_duration)
                    else:
                        self.fixation_start_time = None
                        self.hypnotized = None

                    # Push the data into the output queue
                    self.output_queue.put({
                        "distracted": self.distracted,
                        "drowsy": self.drowsy_bool,
                        "hypnotized": self.hypnotized,
                        "yaw": self.avg_yaw,
                        "pitch": self.avg_pitch,
                        "fixation_start_time":(time.time() - self.fixation_start_time)/ 3,
                        "distracted_duration":(time.time() - (self.distracted_start_time if self.distracted_start_time != None else 0))/3,
                        "drowsy_confidence":self.drowsy_confidence
                    })

    def run(self):
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
            self.process_frame(frame, camera_matrix)

        cap.release()
