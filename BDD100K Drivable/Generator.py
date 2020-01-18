import cv2 as cv
import numpy as np


def Main(img1, img2):
    _, height, width, channel = img1.shape
    img_result, label_result = np.zeros_like(img1), np.zeros_like(img2)
    rotation = 10

    for n, img_file in enumerate(img1):
        # 랜덤 회전
        Rotation = np.random.randint(rotation)
        if np.random.rand(1) < 0.5:
            Rotation = -Rotation

        # 랜덤 반전
        if np.random.rand(1) < 0.5:
            Flip = True
        else:
            Flip = False

        label_file = img2[n]

        if Flip:
            img_file = cv.flip(img_file, 1)
            label_file = cv.flip(label_file, 1)

        M = cv.getRotationMatrix2D((height // 2, width // 2), Rotation, 1)
        img_result[n, :, :, :] = cv.warpAffine(img_file, M, (width, height))
        label_result[n, :, :, :] = cv.warpAffine(label_file, M, (width, height))

    return img_result, label_result
