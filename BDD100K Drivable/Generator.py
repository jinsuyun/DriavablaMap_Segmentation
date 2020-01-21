import cv2 as cv
import numpy as np


def Main(img1, img2):
    _, height, width, channel = img1.shape
    img_result, label_result = [], []
    h_center, w_center = height // 2, width // 2
    rotation = 10

    for n, img_file in enumerate(img1):
        # 랜덤 회전
        Rotation = np.random.rand(1) * rotation
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

        M = cv.getRotationMatrix2D((h_center, w_center), Rotation, 1)
        img_file = (cv.warpAffine(img_file, M, (width, height)))
        label_file = (cv.warpAffine(label_file, M, (width, height)))

        img_result.append(img_file)
        label_result.append(label_file)

        concat = cv.hconcat([img_file, label_file])
        font = cv.FONT_HERSHEY_PLAIN
        cv.putText(concat, str(Rotation) + str(Flip), (w_center * 2, h_center), font, 2, (255, 255, 255))
        cv.imshow('Generator', concat)
        cv.waitKey(1)

    cv.destroyAllWindows()
    img_result, label_result = np.array(img_result), np.array(label_result)

    return img_result, label_result
