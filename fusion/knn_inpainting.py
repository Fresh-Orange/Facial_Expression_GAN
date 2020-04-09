import cv2
import numpy as np
import os
from tqdm import tqdm

def KNN_inpaint(img, ksize):
    h, w, c = np.shape(img)
    step = max(1, (h+w) // 100)
    print("step ", step)

    ret = np.copy(img)

    for i in tqdm(range(h)):
        for j in range(w):
            if np.mean(img[i, j, :]) > 0:
                continue
            candi = []

            def check_pixel(x, y):
                if 0 <= x < h and 0 <= y < w and np.mean(img[x, y, :]) > 0:
                    weight = np.sqrt(((x - i) ** 2 + (y - j) ** 2))
                    candi.append((weight, img[x, y, :]))

            l = 1
            while (len(candi) < ksize):
                for x in range(max(0, i - l), min(h, i + l + 1)):
                    check_pixel(x, j - l)
                    check_pixel(x, j + l)
                for y in range(max(0, j - l + 1), min(w, j + l)):  # avoid repeat count corner
                    check_pixel(i - l, y)
                    check_pixel(i + l, y)
                l += step

            if len(candi) > 0:
                sorted_candi = sorted(candi, key=lambda m: np.mean(m[1]))
                #sorted_candi = sorted(candi, key=lambda m: m[0])
                sorted_candi = sorted_candi[5:]
                p = np.zeros_like(ret[i, j, :]).astype(np.float32)
                weights = 0
                for x in range(min(len(sorted_candi), ksize)):
                    weights += 1 / sorted_candi[x][0]
                    p += sorted_candi[x][1] / sorted_candi[x][0]
                ret[i, j, :] = p / weights
    return ret

def KNN_inpaint_with_mask(img, ksize, mask):
    h, w, c = np.shape(img)
    assert np.shape(img) == np.shape(mask)
    step = max(1, (h+w) // 100)
    print("step ", step)

    ret = np.copy(img)

    for i in tqdm(range(h)):
        for j in range(w):
            if np.mean(mask[i, j, :]) > 0:
                continue
            candi = []

            def check_pixel(x, y):
                if 0 <= x < h and 0 <= y < w and np.mean(mask[x, y, :]) > 0:
                    weight = np.sqrt(((x - i) ** 2 + (y - j) ** 2))
                    candi.append((weight, img[x, y, :]))

            l = 1
            while (len(candi) < ksize):
                for x in range(max(0, i - l), min(h, i + l + 1)):
                    check_pixel(x, j - l)
                    check_pixel(x, j + l)
                for y in range(max(0, j - l + 1), min(w, j + l)):  # avoid repeat count corner
                    check_pixel(i - l, y)
                    check_pixel(i + l, y)
                l += step

            if len(candi) > 0:
                sorted_candi = sorted(candi, key=lambda m: np.mean(m[1]))
                #sorted_candi = sorted(candi, key=lambda m: m[0])
                #sorted_candi = sorted_candi[5:]
                p = np.zeros_like(ret[i, j, :]).astype(np.float32)
                weights = 0
                for x in range(min(len(sorted_candi), ksize)):
                    weights += 1 / sorted_candi[x][0]
                    p += sorted_candi[x][1] / sorted_candi[x][0]
                ret[i, j, :] = p / weights
    return ret

def inpainting(img, mask):
    h, w, c = np.shape(img)
    small_img = cv2.resize(img, (w // 16, h // 16))
    small_mask = cv2.resize(mask, (w // 16, h // 16))
    inpainting = KNN_inpaint_with_mask(small_img, 10, small_mask)
    inpainting = cv2.resize(inpainting, (w, h))
    inpainting[:, :, 0] = np.where(img[:, :, 0] > 0, img[:, :, 0], inpainting[:, :, 0])
    inpainting[:, :, 1] = np.where(img[:, :, 1] > 0, img[:, :, 1], inpainting[:, :, 1])
    inpainting[:, :, 2] = np.where(img[:, :, 2] > 0, img[:, :, 2], inpainting[:, :, 2])
    return inpainting

if __name__ == "__main__":
    input_dir = "examples/try_padding"
    for i in os.listdir(input_dir):
        print(i)
        f = os.path.join(input_dir, i)
        img = cv2.imread(f)
        h, w, c = np.shape(img)
        small_img = cv2.resize(img, (w // 16, h // 16))
        inpainting = KNN_inpaint(small_img, 4)
        inpainting = cv2.resize(inpainting, (w, h))
        inpainting[:,:, 0] = np.where(img[:,:, 0] > 0, img[:,:,0], inpainting[:,:,0])
        inpainting[:,:, 1] = np.where(img[:,:, 1] > 0, img[:,:,1], inpainting[:,:,1])
        inpainting[:,:, 2] = np.where(img[:,:, 2] > 0, img[:,:,2], inpainting[:,:,2])
        cv2.imwrite("examples/try_padding/knn_{}".format(i), inpainting)

