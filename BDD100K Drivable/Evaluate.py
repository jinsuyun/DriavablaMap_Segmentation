from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2 as cv
import glob
import Model
import Load
import Generator

ds = 'D:/Dataset/bdd100k/images/100k/test/'
dirs = glob.glob(ds + '*.jpg')
np.random.shuffle(dirs)
img = Load.LoadImg(dirs, 2, False)
(trimg, trlabel), (teimg, telabel) = Load.Main()
img1, img2 = Generator.Main(trimg, trlabel)
print(img1.shape)
cv.destroyAllWindows()
