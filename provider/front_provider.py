import cv2
import numpy as np


# Thicken a line via the cv2 package
def thicken_front(mask, thickness=5):
    kernel = np.ones((thickness, thickness), np.uint8)

    mask_pad_line = cv2.dilate(mask, kernel, iterations=1)

    return mask_pad_line
