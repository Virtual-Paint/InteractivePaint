import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np
import cv2

from .utils import draw_landmarks_on_image
from ImageProcessing.utils import DrawingSettings
from ImageProcessing.GesturesRecognition.recognize import Recognizer


class LandmarkDetection:
    MODEL_PATH = 'ImageProcessing/MediaPipe/models/hand_landmarker.task'

    def __init__(self, shape: tuple):
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.recognizer = Recognizer()

        self.previous_position = None
        self.shape = shape

    def process_image(self, image: Image, sketch: np.ndarray, drawing_setup: DrawingSettings) -> Image:
        numpy_image = np.asarray(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        detection_result = self.detector.detect(mp_image)

        if detection_result.hand_landmarks:
            gesture = self.recognizer.recognize_gesture(detection_result)
            if gesture == 'finger':
                self._draw_on_sketch(detection_result, sketch, drawing_setup)
            else:
                self.previous_position = None
        else:
            self.previous_position = None
        annotated_image = draw_landmarks_on_image(numpy_image, detection_result)
        
        return Image.fromarray(annotated_image), sketch
    
    def _draw_on_sketch(self, detection_result, sketch: np.ndarray, drawing_setup: DrawingSettings) -> None:
        hand_landmarks_list = detection_result.hand_landmarks[0]

        pointing_finger = hand_landmarks_list[7]
        
        denormalized_coordinates = (int(pointing_finger.x * self.shape[1]), int(pointing_finger.y * self.shape[0]))
 
        if self.previous_position:
            cv2.line(sketch, self.previous_position, denormalized_coordinates, drawing_setup.color, drawing_setup.thickness)
        self.previous_position = denormalized_coordinates


# def main():
#     detection = LandmarkDetection((480, 640))
#     test = Image.open('test.jpg')
#     sketch = np.zeros((480, 640, 3), np.uint8)
    
#     detection.process_image(test, sketch)


# main()