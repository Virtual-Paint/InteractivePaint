from PIL import Image
from io import BytesIO
from dataclasses import dataclass
from enum import Enum
import base64
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from collections import deque


class CustomDeque(deque):
    def is_last_3_gestures_same(self) -> bool:
        return len(set(list(self)[-3:])) == 1
    
    def draw_shape(self, gesture: str) -> bool:
        return len(set(self)) == 1 and self[0] == gesture
    
    def create_sketch_copy(self, gesture: str) -> bool:
        return self.is_last_3_gestures_same() and self[-1] == gesture and self[0] != gesture
    
    def perform_action(self):
        return self.is_last_3_gestures_same() and self[0] != self[-1]
    
    
class Coordinates:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        
    def __iter__(self):
        yield self.x
        yield self.y


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
    

def convert_from_bytes(bytes: str) -> Image:
    base64_data = bytes.split(',')[1]
    binary_data = base64.b64decode(base64_data)
    image_stream = BytesIO(binary_data)
    image = Image.open(image_stream)
    return image
    
def convert_to_bytes(image: Image) -> str:
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    base64_image = f"{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    return base64_image


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image: Image, hand_landmarks: list) -> Image:
    if not hand_landmarks:
        return rgb_image
    annotated_image = np.copy(rgb_image)

    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

    return Image.fromarray(annotated_image)
