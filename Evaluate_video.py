import numpy as np
import cv2 as cv
import Model

video_path = 'D:/20200124_164253.mp4'
result_save1 = 'D:/Predict.mp4'
result_save2 = 'D:/Mask.mp4'
model, epoch = Model.LoadSavedModel(answer='\n')
vid = cv.VideoCapture(video_path)

codec = cv.VideoWriter_fourcc(*'mp4v')
result1 = cv.VideoWriter(result_save1, codec, 30, (512, 288), True)
result2 = cv.VideoWriter(result_save2, codec, 30, (512, 288), False)

threshold = 0.8
while vid.isOpened():
    load, image = vid.read()
    if load:
        img = cv.resize(image, (512, 288))
        predict = np.reshape(model.predict(np.expand_dims(img, axis=0) / 255), [288, 512, 3]) * 255
        predict[predict < (255 * threshold)] = 0
        predict[:, :, 1] = 0
        imgpred = cv.add(img, predict, dtype=cv.CV_8U)

        lane_mask = cv.cvtColor(predict, cv.COLOR_BGR2GRAY)
        kernel = np.ones([8, 8]) / 64
        lane_mask = cv.filter2D(lane_mask, -1, kernel)
        img[lane_mask == 0] = 0
        blur_img = cv.GaussianBlur(img, (3, 3), 0)
        gray_img = cv.cvtColor(blur_img, cv.COLOR_BGR2GRAY)

        result1.write(imgpred)
        result2.write(gray_img)
        cv.imshow('imgpred', imgpred)
        cv.imshow('mask img', gray_img)
        cv.waitKey(100)
    else:
        break

filename = result_save1.split('/')[-1].split('.')[0]
filepath = result_save1[:result_save1.find(filename)]
print('finished. Please check result on', filepath)

vid.release()
result1.release()
result2.release()
