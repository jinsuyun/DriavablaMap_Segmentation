# from progress.bar import IncrementalBar as Bar
# -*- coding: utf-8 -*-
# from darkflow.net.build import TFNet
import glob
import cv2
import os
import argparse
import Model

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--gpus', default='3', type=str, help='Which GPUs you want to use? (0,1,2,3)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

model = Model.SegModel(3)
model.load('\n')
threshold = 0.8

# for BDD100k

def pred():
    imgs = glob.glob('bdd100k/images/100k/test/*')
    # imgs = glob.glob('kookmin_data/data/image/*')
    # imgs = glob.glob('GTA/01_images/*')
    threshold = 0.8

    # predict img save
    for path in sorted(imgs):
        img = cv2.imread(path)
        img = cv2.resize(img, (512, 288), interpolation=cv2.INTER_CUBIC)

        predict = np.reshape(model.predict(np.expand_dims(img, axis=0) / 255), [288, 512, 3]) * 255
        predict[predict < (255 * threshold)] = 0
        predict[:, :, 1] = 0
        imgpred = cv2.add(img, predict, dtype=cv2.CV_8U)
        # cv2.imwrite('result/BDD100k/' + path.split('/')[-1], imgpred)
        # cv2.imwrite('result/kookmin/' + path.split('/')[-1], imgpred)
        # cv2.imwrite('result/GTA/01_images/' + path.split('/')[-1], imgpred)

        # cv2.imwrite('result/BDD100k/baseline/' + path.split('/')[-1], imgpred)
        cv2.imwrite('result/BDD100k/student/' + path.split('/')[-1], imgpred)
        # cv2.imwrite('result/GTA/baseline/' + path.split('/')[-1], imgpred)
        # cv2.imwrite('result/GTA/student/' + path.split('/')[-1], imgpred)
    print("Predict COMPLETED !!!!")

def img2video():
    # image_folder = 'result/BDD100k/'
    # image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))
    image_folder = 'result/GTA/01_images/'
    image_paths = glob.glob(os.path.join(image_folder, '*.png'))

    width = 512
    height = 288
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter("result/BDD100k_pred.avi", fourcc, 1, size)
    out = cv2.VideoWriter("result/GTA_01_pred.avi", fourcc, 14, size)

    for file in sorted(image_paths):
        img = cv2.imread(file)
        img = cv2.resize(img, (512, 288), interpolation=cv2.INTER_CUBIC)
        out.write(img)
    out.release()
    print("Convert COMPLETED !!!!")

# for KODAS

# def pred():
#     imgs = glob.glob('KODAS1/Input/*')
#     threshold = 0.8
#
#     # predict img save
#     for path in sorted(imgs):
#         img = cv2.imread(path)
#         img = cv2.resize(img, (1920, 1200), interpolation=cv2.INTER_CUBIC)
#
#         predict = np.reshape(model.predict(np.expand_dims(img, axis=0) / 255), [1200, 1920, 3]) * 255
#         predict[predict < (255 * threshold)] = 0
#         predict[:, :, 1] = 0
#         imgpred = cv2.add(img, predict, dtype=cv2.CV_8U)
#         cv2.imwrite('result/KODAS/' + path.split('/')[-1], imgpred)
#     print("Predict COMPLETED !!!!")
#
# def img2video():
#     image_folder = 'result/KODAS/'
#     image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))
#
#     width = 1920
#     height = 1200
#     size = (width, height)
#
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter("result/KODAS_pred.avi", fourcc, 25, size)
#
#     for file in sorted(image_paths):
#         img = cv2.imread(file)
#         img = cv2.resize(img, (1920, 1200), interpolation=cv2.INTER_CUBIC)
#         out.write(img)
#     out.release()
#     print("Convert COMPLETED !!!!")

if __name__ == "__main__":
    pred()
    # img2video()