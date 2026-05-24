import numpy as np
from PIL import Image
import open_clip
import torch

class ImageEmbedder:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def embed_image(self, pil_image: Image.Image) -> np.ndarray:
        img = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img)
            image_features = image_features.cpu().numpy()
        return image_features[0]

    def embed_images(self, pil_images: list) -> np.ndarray:
        return np.stack([self.embed_image(img) for img in pil_images])

# Example usage:
# from PIL import Image
# embedder = ImageEmbedder()
# img = Image.open("path_to_image.jpg")
# vec = embedder.embed_image(img)
