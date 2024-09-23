## Get Started

1. python -m env venv
2. Activate env
   2.1 source venv/bin/activate # On macOS/Linux
   2.2 .\venv\Scripts\activate # On Windows
3. pip install -r requirements.txt
4. Download shape predictor
   4.1 https://www.kaggle.com/datasets/sergiovirahonda/shape-predictor-68-face-landmarksdat?resource=download
5. Place it inside a folder called "predictions"
6. python main.py

### installs:

- opencv-python
- mediapipe
- dlib (maybe remove)
- scipy ?

Use pip freeze > requirements.txt to add more necessary depenencies to requirements.txt

### Tech Stack and Libraries for Implementation

- OpenCV: For real-time camera feed and facial landmark detection.
- Mediapipe (Google): For facial landmark tracking (for eyes, mouth, head pose).
- TensorFlow/PyTorch: For training or using pre-trained models to classify expressions of drowsiness.

### Thanks for the Icons!

<a href="https://www.flaticon.com/free-icons/engine" title="engine icons">Engine icons created by juicy_fish - Flaticon</a>
