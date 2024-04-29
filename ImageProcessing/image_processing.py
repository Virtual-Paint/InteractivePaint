from PIL import Image
import numpy as np

from ImageProcessing.MediaPipe.gesture_detector import GestureDetector
from .utils import convert_from_bytes, convert_to_bytes, draw_landmarks_on_image
from ImageProcessing.KalmanFilter.kalman import KalmanFilter
from ImageProcessing.GAN.inpainter import Inpainter
from ImageProcessing.sketch_data import Sketch
from models import InpaintModel


class ImageProcessing:
    def __init__(self):
        self.gesture_detector = GestureDetector()
        self.kalman = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1) 
        self.inpainter = Inpainter()
        

    def process_image(self, bytes: str, sketch: Sketch) -> str:
        image = convert_from_bytes(bytes)

        gesture, hand_landmarks = self.gesture_detector.process_image(image)
        
        if gesture:
            sketch.perform_action(gesture, hand_landmarks)

        return {
            'processed_input': self._process_input_image(image, hand_landmarks),
            'sketch': sketch.get_bytes_sketch(),
            'color': sketch.color.name,
            'thickness': sketch.thickness.name
        }
        
    def inpaint_sketch(self, body: InpaintModel) -> str:
        return self.inpainter.process_sketch(body)
    
    @staticmethod
    def _process_input_image(image: Image, hand_landmarks: list) -> str:
        processed_input = draw_landmarks_on_image(image, hand_landmarks)
        return convert_to_bytes(processed_input)
