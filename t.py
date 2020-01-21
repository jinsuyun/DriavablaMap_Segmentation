import glob
import numpy as np
import cv2 as cv

path = glob.glob('D:/Dataset/bdd100k/images/100k/test/*.jpg')
np.random.shuffle(path)

print(path[0])
array = []
for n, file in enumerate(path):
    img = cv.imread(file, cv.IMREAD_REDUCED_COLOR_4)
    cv.imshow('loaded', img)
    cv.waitKey(1)
    array.append(img)
    if n + 1 == 10:
        array = np.array(array)
        cv.destroyAllWindows()
        break

print(array.shape)

for file in array:
    cv.imshow('eval', file)
    cv.waitKey(0)
