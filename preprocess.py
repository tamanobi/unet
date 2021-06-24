import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path
from PIL import Image


def save_as_indexed_png(filepath: Path, gray):
    palette = sum([
        [0, 0, 0],
        [255, 255, 255]]
    , [])

    with Image.fromarray(gray, mode="P") as img:
        img.putpalette(palette)
        img.save(str(filepath))

d = Path("data/SegmentationClass")
out_d = Path("zero_one")
if not out_d.exists():
    out_d.mkdir()
for path in d.glob("*.png"):
    img = image.load_img(str(path))
    arr = image.img_to_array(img)
    white = np.array([0, 0, 0], dtype=np.uint8)
    one = np.array([1, 1, 1], dtype=np.uint8)
    print(arr.shape)
    print(arr[0, 0, :])
    output = arr.copy()
    valid = np.all(arr == [255, 0, 124], axis=-1)
    print(np.sum(valid))
    rs, cs = valid.nonzero()
    output[rs, cs, :] = [1, 1, 1]
    save_as_indexed_png(out_d / path.name, output[:, :, 0])
