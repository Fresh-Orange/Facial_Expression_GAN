import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

def psnr(a, b):
    origin = cv2.imread(a, cv2.IMREAD_COLOR)
    first = cv2.imread(b, cv2.IMREAD_COLOR)
    ssim_origin = peak_signal_noise_ratio(origin, first)
    return ssim_origin

if __name__ == '__main__':
    a = "images/11085.jpg"
    b = "images/11085_nomatting.jpg"
    c = "images/11085_ssim15W_matting.jpg"
    print(psnr(a, b))
    print(psnr(a, c))