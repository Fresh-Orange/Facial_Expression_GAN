import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

def ssim(a, b):
    origin = cv2.imread(a, cv2.IMREAD_COLOR)
    first = cv2.imread(b, cv2.IMREAD_COLOR)
    ssim_origin = structural_similarity(origin, first, multichannel=True)
    return ssim_origin

if __name__ == '__main__':
    a = "images/11085.jpg"
    b = "images/11085_nomatting.jpg"
    c = "images/11085_ssim15W_matting.jpg"
    print(ssim(a, b))
    print(ssim(a, c))

