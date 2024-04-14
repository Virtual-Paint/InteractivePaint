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
        self.GESTURE_ACTION = {
            'one': self._draw,
            'stop': self._rubber,
            'four': self._change_color,
            'three2': self._change_thickness,
            'peace': self._draw_circle
        }

    def process_image(self, image: Image, sketch: np.ndarray, drawing_setup: DrawingSettings) -> Image:
        numpy_image = np.asarray(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        detection_result = self.detector.detect(mp_image)

        if detection_result.hand_landmarks:     #TODO przerobic ten element
            gesture = self.recognizer.recognize_gesture(detection_result)
            hand_landmarks_list = detection_result.hand_landmarks[0]
            if gesture == 'finger':
                self._draw(hand_landmarks_list, sketch, drawing_setup)
            elif gesture == 'palm':
                self._rubber(hand_landmarks_list, sketch)
            elif gesture == 'peace':     # TODO dodać więcej kolorów, aby zmieniały się na następny po kazdym wykryciu gestu
                drawing_setup.change_color(drawing_setup)
            # elif gesture == 'peace':
            #     drawing_setup.thickness = 8
            else:
                self.previous_position = None
        else:
            self.previous_position = None
        annotated_image = draw_landmarks_on_image(numpy_image, detection_result)
        
        return Image.fromarray(annotated_image), sketch
    
    def _draw(self, hand_landmarks_list: list, sketch: np.ndarray, drawing_setup: DrawingSettings) -> None:
        pointing_finger = hand_landmarks_list[7]
        
        denormalized_coordinates = (int(pointing_finger.x * self.shape[1]), int(pointing_finger.y * self.shape[0]))
 
        if self.previous_position:
            cv2.line(sketch, self.previous_position, denormalized_coordinates, drawing_setup.color.value(), drawing_setup.thickness.value())
        self.previous_position = denormalized_coordinates

    def _rubber(self, hand_landmarks_list: list, sketch: np.ndarray) -> None:
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
        
    def _change_color(self, drawing_setup: DrawingSettings):
        drawing_setup.change_color()
    
    def _change_thickness(self, drawing_setup: DrawingSettings):
        drawing_setup.change_thickness()
    
    def _draw_circle(self):
        raise NotImplementedError
