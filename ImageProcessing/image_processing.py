from PIL import Image
from io import BytesIO
import base64
import numpy as np

from ImageProcessing.MediaPipe.finger_tracker import LandmarkDetection


class ImageProcessing:
    def __init__(self):
        self.sketch_shape = (480, 640)      #TODO - do env? na podstawie przychodzącego obrazu?
        self.sketch = np.zeros((*self.sketch_shape, 3), np.uint8)
        
        self.landmark_detector = LandmarkDetection()

    async def process_image(self, bytes: str) -> str:
        image = self._convert_from_bytes(bytes)

        processed_input = self.landmark_detector.process_image(image)

        sketch = Image.fromarray(self.sketch)
        sketch = self._convert_to_bytes(sketch)

        processed_input = self._convert_to_bytes(processed_input)
        return {
            'processed_input': processed_input,
            'sketch': sketch
        }

    @staticmethod
    def _convert_from_bytes(bytes: str) -> Image:
        base64_data = bytes.split(',')[1]
        binary_data = base64.b64decode(base64_data)
        image_stream = BytesIO(binary_data)
        image = Image.open(image_stream)
        return image
    
    @staticmethod
    def _convert_to_bytes(image: Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        base64_image = f"{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
        return base64_image
