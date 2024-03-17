from PIL import Image
from io import BytesIO
import base64


class ImageProcessing:
    def __init__(self):
        pass

    def process_image(self, bytes: str) -> str:
        image = self._convert_from_bytes(bytes)
        return self._convert_to_bytes(image)

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
