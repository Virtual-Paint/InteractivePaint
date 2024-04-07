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

        if detection_result.hand_landmarks:     #TODO przerobic ten element
            gesture = self.recognizer.recognize_gesture(detection_result)
            if gesture == 'finger':
                self._draw(detection_result, sketch, drawing_setup)
            elif gesture == 'palm':
                self._rubber(detection_result, sketch, drawing_setup)
            elif gesture == 'fist':     # TODO dodać więcej kolorów, aby zmieniały się na następny po kazdym wykryciu gestu
                drawing_setup.color = (255, 0, 0)
            else:
                self.previous_position = None
        else:
            self.previous_position = None
        annotated_image = draw_landmarks_on_image(numpy_image, detection_result)
        
        return Image.fromarray(annotated_image), sketch
    
    def _draw(self, detection_result, sketch: np.ndarray, drawing_setup: DrawingSettings) -> None:
        hand_landmarks_list = detection_result.hand_landmarks[0]

        pointing_finger = hand_landmarks_list[7]
        
        denormalized_coordinates = (int(pointing_finger.x * self.shape[1]), int(pointing_finger.y * self.shape[0]))
 
        if self.previous_position:
            cv2.line(sketch, self.previous_position, denormalized_coordinates, drawing_setup.color, drawing_setup.thickness)
        self.previous_position = denormalized_coordinates

    def _rubber(self, detection_result, sketch: np.ndarray, drawing_setup: DrawingSettings) -> None:
        hand_landmarks_list = detection_result.hand_landmarks[0]

        wrist = hand_landmarks_list[0]
        index_finger_mcp = hand_landmarks_list[5]
        pinky_mcp = hand_landmarks_list[17]

        denormalized_wrist = (int(wrist.x * self.shape[1]), int(wrist.y * self.shape[0]))
        denormalized_index_finger_mcp = (int(index_finger_mcp.x * self.shape[1]), int(index_finger_mcp.y * self.shape[0]))
        denormalized_pinky_mcp = (int(pinky_mcp.x * self.shape[1]), int(pinky_mcp.y * self.shape[0]))

        x_center = int((denormalized_wrist[0] + denormalized_index_finger_mcp[0] + denormalized_pinky_mcp[0]) / 3)
        y_center = int((denormalized_wrist[1] + denormalized_index_finger_mcp[1] + denormalized_pinky_mcp[1]) / 3)

        radius = 10
        color = (255, 255, 255)
        thickness = -1
        cv2.circle(sketch, (x_center, y_center), radius, color, thickness)


# def main():
#     detection = LandmarkDetection((480, 640))
#     test = Image.open('test.jpg')
#     sketch = np.zeros((480, 640, 3), np.uint8)
    
#     detection.process_image(test, sketch)


# main()