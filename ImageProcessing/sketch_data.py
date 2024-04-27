import numpy as np
import cv2
from PIL import Image
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from .utils import Colors, Thickness, convert_to_bytes, CustomDeque, Coordinates
from ImageProcessing.KalmanFilter.kalman import KalmanFilter


class Sketch:
    def __init__(self, kalman: KalmanFilter) -> None:
        self.kalman = kalman
        
        self.shape = (480, 640)      #TODO - do env? na podstawie przychodzącego obrazu?
        self.sketch = np.zeros((*self.shape, 3), np.uint8) + 255     #TODO to i shape do drawing_setup ??
        self.sketch_history = None       
        
        self.color = Colors.BLACK
        self.thickness = Thickness.MEDIUM
        
        self.gestures_log = CustomDeque([None]*4, maxlen=4)
        self.previous_position = None
        self.prev_pos_for_shapes = None
        
        self.mapping = {
            'ONE': self._draw,
            'STOP': self._rubber,
            'PEACE': self._draw_shape,
            'ROCK': self._draw_shape,
            'FOUR': self._change_color,
            'THREE2': self._change_thickness
        }
        
    def perform_action(self, gesture: str, hand_landmarks: list) -> None:
        self.gestures_log.append(gesture)
        
        self._calculate_kalman(hand_landmarks)
        
        self._check_initial_conditions()
        
        if gesture in self.mapping:
            self.mapping[gesture](hand_landmarks_list=hand_landmarks)
    
    def get_bytes_sketch(self) -> str:
        sketch = Image.fromarray(self.sketch)
        return convert_to_bytes(sketch)
    
    def _check_initial_conditions(self) -> None:
        if self.gestures_log.clear_prev_pos():
            self.previous_position = None
        if self.gestures_log.clear_shape_prev_pos():
            self.prev_pos_for_shapes = None
        if self.gestures_log.perform_action():
            self.sketch_history = np.copy(self.sketch)
    
    def _calculate_kalman(self, hand_landmarks_list: list) -> None:
        index_pos = self._denormalize_coordinates(hand_landmarks_list[8])
        center = np.matrix([[index_pos.x],
                            [index_pos.y]])
        
        prediction, self.estimation = self.kalman.calculate(center)
        
    def _draw(self, **kwargs) -> None:
        if self.previous_position:
            cv2.line(self.sketch, self.previous_position, self.estimation, self.color.value, self.thickness.value)
        self.previous_position = self.estimation
            
    def _draw_shape(self, hand_landmarks_list: list, **kwargs) -> None:
        gesture = self.gestures_log[-1]
        if not self.gestures_log.draw_shape(gesture):
            return
        
        finger_1, finger_2 = self._denormalize_coordinates(hand_landmarks_list[8], hand_landmarks_list[12])
        
        coordinates = Coordinates(int((finger_1.x + finger_2.x) / 2),
                                  int((finger_1.y + finger_2.y) / 2))
        
        if self.prev_pos_for_shapes:
            self.sketch = np.copy(self.sketch_history)
            if gesture == 'PEACE':
                cv2.rectangle(self.sketch, tuple(self.prev_pos_for_shapes), tuple(coordinates), self.color.value, self.thickness.value)
            elif gesture == 'ROCK':
                radius = int(abs(coordinates.y - self.prev_pos_for_shapes.y))
                cv2.circle(self.sketch, tuple(self.prev_pos_for_shapes), radius, self.color.value, self.thickness.value)
        else:
            self.prev_pos_for_shapes = coordinates

    def _rubber(self, hand_landmarks_list: list, **kwargs) -> None:
        wrist_pos, index_pos, pinky_pos = self._denormalize_coordinates(hand_landmarks_list[0], 
                                                                        hand_landmarks_list[5], 
                                                                        hand_landmarks_list[17])

        coordinates = Coordinates(int((wrist_pos.x + index_pos.x + pinky_pos.x) / 3),
                                  int((wrist_pos.y + index_pos.y + pinky_pos.y) / 3))
        
        cv2.circle(self.sketch, tuple(coordinates), 10, (255, 255, 255), -1)
        
    def _change_color(self, **kwargs) -> None:
        if self.gestures_log.perform_action():
            colors = list(Colors)
            idx = colors.index(self.color) + 1
            self.color = colors[idx] if idx < len(Colors) else colors[0]
            print(f'Changed color! New color is {self.color}')
        
    def _change_thickness(self, **kwargs) -> None:
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
