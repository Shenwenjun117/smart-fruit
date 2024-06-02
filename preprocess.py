from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset



class SquarePad:
    def __call__(self, image):
        max_size = max(image.size)
        left, top = (max_size - image.size[0]) // 2, (max_size - image.size[1]) // 2
        padding = (left, top, max_size - image.size[0] - left, max_size - image.size[1] - top)
        return ImageOps.expand(image, padding, fill='white')