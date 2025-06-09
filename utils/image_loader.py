from PIL import Image
import numpy as np

def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32)
