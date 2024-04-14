import numpy as np
import cv2
from PIL import Image

from .utils import Colors, Thickness, convert_to_bytes


class Sketch:
    def __init__(self) -> None:
        self.shape = (480, 640)      #TODO - do env? na podstawie przychodzącego obrazu?
        self.sketch = np.zeros((*self.shape, 3), np.uint8) + 255     #TODO to i shape do drawing_setup ??
        
        self.color = Colors.BLACK
        self.thickness = Thickness.MEDIUM
        self.gestures_log = []
        self.previous_position = None
        
    def perform_action(self, gesture: str, hand_landmarks: list) -> None:
        self.gestures_log.append(gesture)
        
        if gesture == 'ONE':
            self._draw(hand_landmarks)
            return
        self.previous_position = None
            
        if len(set(self.gestures_log[:-3])) != 1:
            return
        
        if gesture == 'STOP':
            self._rubber(hand_landmarks)
        elif gesture == 'PEACE':
            self._change_color()
        elif gesture == 'THREE2':
            self._change_thickness()
    
    def get_bytes_sketch(self) -> str:
        sketch = Image.fromarray(self.sketch)
        return convert_to_bytes(sketch)
        
    def _draw(self, hand_landmarks_list: list) -> None:
        pointing_finger = hand_landmarks_list[7]
        
        denormalized_coordinates = (int(pointing_finger.x * self.shape[1]), int(pointing_finger.y * self.shape[0]))
 
        if self.previous_position:
            cv2.line(self.sketch, self.previous_position, denormalized_coordinates, self.color.value, self.thickness.value)
        self.previous_position = denormalized_coordinates

    def _rubber(self, hand_landmarks_list: list) -> None:
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
        cv2.circle(self.sketch, (x_center, y_center), radius, color, thickness)
        
    def _change_color(self) -> None:
        colors = list(Colors)
        idx = colors.index(self.color) + 1
        self.color = colors[idx] if idx < len(Colors) else colors[0]
        
    def _change_thickness(self) -> None:
        thicnesses = list(Thickness)
        idx = thicnesses.index(self.thickness) + 1
        self.thickness = thicnesses[idx] if idx < len(Thickness) else thicnesses[0]
    
    def _draw_circle(self) -> None:
        raise NotImplementedError