import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path
from typing import Generator, Tuple, Protocol

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


for img, mask in image_pairs(config):
    print(img, mask)


def generator(datum):
    for data in datum:
        yield {data[key] for key in ["image", "segmentation_mask"]}


def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  return input_image, input_mask

