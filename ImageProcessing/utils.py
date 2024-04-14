from PIL import Image
from io import BytesIO
from dataclasses import dataclass
from enum import Enum
import base64


class Colors(Enum):
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    
class Thickness(Enum):
    TINY = 2
    MEDIUM = 4
    LARGE = 8
    MASSIVE = 16
    
    
@dataclass
class DrawingSettings:
    color: Colors = Colors.BLACK
    thickness: Thickness = Thickness.MEDIUM
    
    def change_color(self):
        colors = list(Colors)
        idx = colors.index(self.color) + 1
        self.color = colors[idx] if idx < len(Colors) else colors[0]
        
    def change_thickness(self):
        thicnesses = list(Thickness)
        idx = thicnesses.indes(self.thickness) + 1
        self.thickness = thicnesses[idx] if idx < len(Thickness) else thicnesses[0]
    

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
