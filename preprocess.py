import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

path = "data/SegmentationClass/nude22_1.png"
img = image.load_img(path)
arr = image.img_to_array(img)
white = np.array([0, 0, 0], dtype=np.uint8)
one = np.array([1, 1, 1], dtype=np.uint8)
print(arr.shape)
print(arr[0, 0, :])
output = arr.copy()
valid = np.all(arr == [255,0,124], axis=-1)
print(np.sum(valid))
rs, cs = valid.nonzero()
print(rs, cs)
output[rs, cs, :] = [1, 1, 1]
print(output[0, 0, :])
white_img = image.array_to_img(output, scale=False)
image.save_img("white_img.png", white_img)
