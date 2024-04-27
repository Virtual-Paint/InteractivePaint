from PIL import Image
import numpy as np

from ImageProcessing.MediaPipe.gesture_detector import GestureDetector
from .utils import convert_from_bytes, convert_to_bytes, draw_landmarks_on_image
from .sketch_data import Sketch


class ImageProcessing:
    def __init__(self):
        self.sketch = Sketch()      #TODO to wywalić gdzieś do main, tak aby każdy user miał własny sketch
        
        self.gesture_detector = GestureDetector()

    def process_image(self, bytes: str) -> str:
        image = convert_from_bytes(bytes)

        gesture, hand_landmarks = self.gesture_detector.process_image(image)
        
        if gesture:
            self.sketch.perform_action(gesture, hand_landmarks)
        
        return {
            'processed_input': self._process_input_image(image, hand_landmarks),
            'sketch': self.sketch.get_bytes_sketch()
        }
    
    @staticmethod
    def _process_input_image(image: Image, hand_landmarks: list) -> str:
        processed_input = draw_landmarks_on_image(image, hand_landmarks)
        return convert_to_bytes(processed_input)
