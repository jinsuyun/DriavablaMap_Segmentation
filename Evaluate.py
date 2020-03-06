import numpy as np
import cv2 as cv
import glob
import Model

model = Model.LoadSavedModel()
imgs = glob.glob('C:/bdd100k/images/100k/test/*')
np.random.shuffle(imgs)

for path in imgs[:10]:
    img = cv.imread(path)
    img = cv.resize(img, (512, 288))
    predict = np.reshape(model.predict(np.expand_dims(img, axis=0) / 255), [288, 512, 3]) * 255
    imgpred = cv.add(img, predict, dtype=cv.CV_8U)
    cv.imshow('imgpred', imgpred)
    cv.imshow('original', img)
    cv.waitKey(5000)
