import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from os import listdir

from .architecture import Generator


class Inpainter:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize((256, 256), antialias=True),
		transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
	device = "cuda" if torch.cuda.is_available() else "cpu"

	def __init__(self):
		self.models = {}
		self._load_models()
	
	def inpaint_image(self, model: str, image: np.ndarray) -> Image:
		image = np.copy(image)
		min_x, min_y, max_x, max_y = self._find_bounding_box(image)
		cropped_image = image[min_y:max_y, min_x:max_x, :]
		cropped_image = self.transform(cropped_image)
		cropped_image = cropped_image.unsqueeze(0).to(self.device)
		if torch.cuda.is_available():
			self.models[model].cuda()
		inpainted = self.models[model](cropped_image)
		inpainted = inpainted * 0.5 + 0.5
		inpainted.detach()
		inpainted = inpainted.squeeze(0).cpu()
		inpainted = np.array(F.to_pil_image(inpainted).resize((max_x-min_x, max_y-min_y)))
		image[min_y:max_y, min_x:max_x, :] = inpainted
		return F.to_pil_image(image)
	
	def _find_bounding_box(self, image: np.ndarray) -> Image:
		min_x, min_y = 1000, 1000
		max_x, max_y = 0, 0
		for x in range(0, image.shape[1]):
			for y in range(0, image.shape[0]):
				if np.any(image[y, x] < 255):
					if min_x > x:
						min_x = x
					if min_y > y:
						min_y = y
					if max_x < x:
						max_x = x
					if max_y < y:
						max_y = y
		return min_x, min_y, max_x, max_y

	
	def _load_models(self) -> None:
		directory = 'ImageProcessing/GAN/models/'
		weights = listdir(directory)		#TODO - zmienić na wczytywanie sciezki z pliku / pobieranie wag z dysku
		for file in weights:
			model = Generator()
			name = file.split('_')[0]
			checkpoint = torch.load(directory + file, map_location=torch.device(self.device))
			model.load_state_dict(checkpoint['generator_state_dict'])
			model.eval()
			self.models[name] = model
