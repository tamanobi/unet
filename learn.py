import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path
from typing import Generator, Tuple

class Config:
    IMAGE_DIR = "data/SegmentationClass"
    MASK_DIR = "zero_one/"

def image_pairs() -> Generator[Tuple[Path, Path], None, None]:
    for path in Path(Config.IMAGE_DIR).iterdir():
        mask = Path(Config.MASK_DIR) / path.name
        assert mask.exists()
        yield path, mask


for img, mask in image_pairs():
    print(img, mask)


def generator(datum):
    for data in datum:
        yield {data[key] for key in ["image", "segmentation_mask"]}


def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  return input_image, input_mask

