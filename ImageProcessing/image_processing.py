from PIL import Image
import numpy as np

from ImageProcessing.MediaPipe.finger_tracker import LandmarkDetection
from .utils import convert_from_bytes, convert_to_bytes, DrawingSettings


class ImageProcessing:
    def __init__(self):
        self.sketch_shape = (480, 640)      #TODO - do env? na podstawie przychodzącego obrazu?
        self.sketch = np.zeros((*self.sketch_shape, 3), np.uint8) + 255     #TODO to i shape do drawing_setup ??
        self.drawing_setup = DrawingSettings()
        
        self.landmark_detector = LandmarkDetection(self.sketch_shape)

    def process_image(self, bytes: str) -> str:
        image = convert_from_bytes(bytes)

        processed_input, self.sketch = self.landmark_detector.process_image(image, self.sketch, self.drawing_setup)

        sketch = Image.fromarray(self.sketch)
        sketch = convert_to_bytes(sketch)

        processed_input = convert_to_bytes(processed_input)
        return {
            'processed_input': processed_input,
            'sketch': sketch
        }
    
    def set_color(self, color: str) -> None:
        self.drawing_setup.color = tuple(color)

    def set_thickness(self, thickness: int) -> None:
        self.drawing_setup.thickness = thickness
