from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2 as cv
import glob
import Model
import Load

ds = 'D:/Dataset/bdd100k/images/100k/test/'
dirs = glob.glob(ds + '*.jpg')
np.random.shuffle(dirs)
Dir = np.expand_dims(dirs[0].replace('\\', '/'), -1)
img = Load.LoadImg(Dir, 1, False)

model = Model.Main(False)
predict = model.predict(img)
predict = np.squeeze(to_categorical(np.argmax(predict, -1), 3), 0)
