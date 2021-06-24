import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path
from typing import Generator, Tuple, Protocol
import numpy as np

class DirPair(Protocol):
    IMAGE_DIR: str
    MASK_DIR: str

class Config(DirPair):
    IMAGE_DIR: str = "data/SegmentationClass"
    MASK_DIR: str = "zero_one/"

class ConfigFactory:
    @classmethod
    def create(cls) -> Config:
        return Config()

config = ConfigFactory.create()

def image_pairs(config: DirPair) -> Generator[Tuple[Path, Path], None, None]:
    for path in Path(config.IMAGE_DIR).iterdir():
        mask = Path(config.MASK_DIR) / path.name
        assert mask.exists()
        yield path, mask


def image_loader(datum: Generator[Tuple[Path, Path], None, None]) -> Generator[dict, None, None]:
    """numpy 形式で画像を読み込む"""
    for img, mask in datum:
        base_img = image.load_img(str(img))
        mask_img = image.load_img(str(mask), grayscale=True)
        yield {
            "image": image.img_to_array(base_img, dtype=np.uint8),
            "mask": image.img_to_array(mask_img, dtype=np.uint8)
        }

image_loader(image_pairs(config))

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  return input_image, input_mask

