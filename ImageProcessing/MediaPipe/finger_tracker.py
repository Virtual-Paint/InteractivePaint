import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image
import numpy as np

from .utils import draw_landmarks_on_image


class LandmarkDetection:
    MODEL_PATH = 'ImageProcessing/MediaPipe/models/hand_landmarker.task'

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                               num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_image(self, image: Image) -> Image:
        numpy_image = np.asarray(image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        detection_result = self.detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(numpy_image, detection_result)
        return Image.fromarray(annotated_image)