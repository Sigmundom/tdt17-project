import cv2
import numpy as np

from config import IM_HEIGHT, IM_WIDTH

image = cv2.imread('data/test/validation/images/Norway_000005.jpg')
# convert BGR to RGB color format
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image_cropped = image[image.shape[0]-1280:,:]
image_resized = image_cropped
image_resized = cv2.resize(image_cropped, (2020, 1280))
print(image_resized.shape)

cv2.imwrite('test.jpg', image_resized)
