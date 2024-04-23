import numpy as np
import cv2
from PIL import Image
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from .utils import Colors, Thickness, convert_to_bytes, CustomDeque, Coordinates
from ImageProcessing.KalmanFilter.kalman import KalmanFilter


class Sketch:
    def __init__(self) -> None:
        self.shape = (480, 640)      #TODO - do env? na podstawie przychodzącego obrazu?
        self.sketch = np.zeros((*self.shape, 3), np.uint8) + 255     #TODO to i shape do drawing_setup ??
        self.sketch_history = None
                
        self.kalman = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)        
        
        self.color = Colors.BLACK
        self.thickness = Thickness.MEDIUM
        self.gestures_log = CustomDeque([None]*4, maxlen=4)
        self.previous_position = None
        
        self.prev_pos_for_shapes = None
        
    def perform_action(self, gesture: str, hand_landmarks: list) -> None:
        self.gestures_log.append(gesture)
        
        self._calculate_kalman(hand_landmarks)
        
        if self.gestures_log[-1] != 'ONE' and self.gestures_log[-2] != 'ONE' and self.gestures_log[-3] != 'ONE':
            self.previous_position = None
        if len(set(list(self.gestures_log)[:3])) != 1 and self.gestures_log[-1] != self.gestures_log[0]:
            self.prev_pos_for_shapes = None
        if self.gestures_log.perform_action():
            self.sketch_history = np.copy(self.sketch)
        
        
        if gesture == 'ONE':
            self._draw()
        elif gesture == 'STOP':
            self._rubber(hand_landmarks)
        elif gesture == 'PEACE':
            self._draw_rectangle(hand_landmarks)   
        elif gesture == 'ROCK':
            self._draw_circle(hand_landmarks)
        elif gesture == 'FOUR':
            self._change_color()
        elif gesture == 'THREE2':
            self._change_thickness()
    
    def get_bytes_sketch(self) -> str:
        sketch = Image.fromarray(self.sketch)
        return convert_to_bytes(sketch)
    
    def _calculate_kalman(self, hand_landmarks_list: list) -> None:
        index_pos = self._denormalize_coordinates(hand_landmarks_list[8])
        center = np.matrix([[index_pos.x],
                            [index_pos.y]])
        
        prediction, self.estimation = self.kalman.calculate(center)
        
    def _draw(self) -> None:
        if self.previous_position:
            cv2.line(self.sketch, self.previous_position, self.estimation, self.color.value, self.thickness.value)
        self.previous_position = self.estimation
    
    def _draw_circle(self, hand_landmarks_list: list) -> None:
        if not self.gestures_log.draw_shape('ROCK'):
            return
        index_pos, pinky_pos = self._denormalize_coordinates(hand_landmarks_list[8], hand_landmarks_list[20])
        
        coordinates = Coordinates(int((index_pos.x + pinky_pos.x) / 2),
                                  int((index_pos.y + pinky_pos.y) / 2))
        
        if self.prev_pos_for_shapes:
            radius = int(abs(coordinates.y - self.prev_pos_for_shapes.y))
            self.sketch = np.copy(self.sketch_history)
            cv2.circle(self.sketch, tuple(self.prev_pos_for_shapes), radius, self.color.value, self.thickness.value)
        else:
            self.prev_pos_for_shapes = coordinates
    
    def _draw_rectangle(self, hand_landmarks_list: list) -> None:
        if not self.gestures_log.draw_shape('PEACE'):
            return
        index_pos, middle_pos = self._denormalize_coordinates(hand_landmarks_list[8], hand_landmarks_list[12])
        
        coordinates = Coordinates(int((index_pos.x + middle_pos.x) / 2),
                                  int((index_pos.y + middle_pos.y) / 2))
        
        if self.prev_pos_for_shapes:
            self.sketch = np.copy(self.sketch_history)
            cv2.rectangle(self.sketch, tuple(self.prev_pos_for_shapes), tuple(coordinates), self.color.value, self.thickness.value)
        else:
            self.prev_pos_for_shapes = coordinates

    def _rubber(self, hand_landmarks_list: list) -> None:
        wrist_pos, index_pos, pinky_pos = self._denormalize_coordinates(hand_landmarks_list[0], 
                                                                        hand_landmarks_list[5], 
                                                                        hand_landmarks_list[17])

        coordinates = Coordinates(int((wrist_pos.x + index_pos.x + pinky_pos.x) / 3),
                                  int((wrist_pos.y + index_pos.y + pinky_pos.y) / 3))
        
        cv2.circle(self.sketch, tuple(coordinates), 10, (255, 255, 255), -1)
        
    def _change_color(self) -> None:
        if self.gestures_log.perform_action():
            colors = list(Colors)
            idx = colors.index(self.color) + 1
            self.color = colors[idx] if idx < len(Colors) else colors[0]
            print(f'Changed color! New color is {self.color}')
        
    def _change_thickness(self) -> None:
        if self.gestures_log.perform_action():
            thicnesses = list(Thickness)
            idx = thicnesses.index(self.thickness) + 1
            self.thickness = thicnesses[idx] if idx < len(Thickness) else thicnesses[0]
            print(f'Changed thickness! New thickness is {self.thickness}')
        
    def _denormalize_coordinates(self, *args: NormalizedLandmark) -> tuple[Coordinates, ...]:
        result = [Coordinates(int(arg.x * self.shape[1]), int(arg.y * self.shape[0])) for arg in args]
        if len(result) == 1:
            return result[0]
        return tuple(result)
