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
		image = self.transform(image)
		image = image.unsqueeze(0).to(self.device)
		if torch.cuda.is_available():
			self.models[model].cuda()
		inpainted = self.models[model](image)
		inpainted = inpainted * 0.5 + 0.5
		inpainted.detach()
		inpainted = inpainted.squeeze(0).cpu()
		return F.to_pil_image(inpainted)
	
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
