from PIL import Image
from io import BytesIO
from dataclasses import dataclass
import base64


@dataclass
class DrawingSettings:
    color: tuple = (0, 0, 0)
    thickness: int = 4
    

def convert_from_bytes(bytes: str) -> Image:
    base64_data = bytes.split(',')[1]
    binary_data = base64.b64decode(base64_data)
    image_stream = BytesIO(binary_data)
    image = Image.open(image_stream)
    return image
    
def convert_to_bytes(image: Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    base64_image = f"{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    return base64_image