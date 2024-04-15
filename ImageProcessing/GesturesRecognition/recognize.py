import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from os import listdir
import cv2

from .architecture import Net
from .utils import LANDMARKS_LINKS, Gestures, ImageShape


class Recognizer:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5], std=[0.5])
	])
	device = "cuda" if torch.cuda.is_available() else "cpu"
	OUTPUT_SHAPE = ImageShape(28, 28)

	def __init__(self):
		self._load_model()
	
	def recognize_gesture(self, landmarks: list) -> Image:
		image = self._convert_to_image(landmarks)
		image = self.transform(image)
		image = image.unsqueeze(0).to(self.device)
		if torch.cuda.is_available():
			self.model.cuda()
		result = self.model(image)
		probability = torch.softmax(result.squeeze(), dim=0)
		gesture = probability.argmax()
		return Gestures(gesture.item()).name
	
	def _convert_to_image(self, landmarks: list) -> np.ndarray:
		x = [landmark.x for landmark in landmarks]
		y = [landmark.y for landmark in landmarks]
		min_x = min(x)
		max_x = max(x)
		min_y = min(y)
		max_y = max(y)
		x_norm = [int((value - min_x) / (max_x - min_x) * (self.OUTPUT_SHAPE.x - 1)) for value in x]
		y_norm = [int((value - min_y) / (max_y - min_y) * (self.OUTPUT_SHAPE.y - 1)) for value in y]

		blank_img = np.zeros(tuple(self.OUTPUT_SHAPE), np.uint8)
		for idx_from, target in LANDMARKS_LINKS.items():
			for idx_to in target:
				blank_img = cv2.line(blank_img, (x_norm[idx_from], y_norm[idx_from]), (x_norm[idx_to], y_norm[idx_to]), (255, 255, 255), 1)
		return blank_img
	
	def _load_model(self) -> None:
		path = 'ImageProcessing/GesturesRecognition/models/hand_recognition_model.pth.tar'
		model = Net(input_shape=1,
			  hidden_units=10,
			  output_shape=7)
		model.load_state_dict(torch.load(path))
		model.eval()
		self.model = model
		