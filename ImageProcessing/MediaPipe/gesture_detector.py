import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
import cv2

from ImageProcessing.sketch_data import Sketch
from ImageProcessing.GesturesRecognition.recognize import Recognizer


class GestureDetector:
    MODEL_PATH = 'ImageProcessing/MediaPipe/models/hand_landmarker.task'

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.recognizer = Recognizer()

    def process_image(self, image: Image) -> tuple[str, list]:
        numpy_image = np.asarray(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            return None, []
        
        hand_landmarks_list = detection_result.hand_landmarks[0]
        gesture = self.recognizer.recognize_gesture(hand_landmarks_list)
        
        return gesture, hand_landmarks_list
