import numpy as np
import cv2 as cv
import glob
import Model
import os
import argparse
import tensorflow as tf
from tensorflow.python.client import device_lib
import warnings
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--gpus', default='3', type=str, help='Which GPUs you want to use? (0,1,2,3)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

print(tf.config.list_physical_devices())
#print(device_lib.list_local_devices())
#if gpus:
#    for gpu in gpus:
#print(device_lib.list_local_devices())

with tf.device('/gpu:0'):
    model = Model.SegModel(3)
    model.load('\n')
    # imgs = glob.glob('bdd100k/images/100k/test/*')
    # imgs = glob.glob('KODAS1/Input/*')
    # imgs = glob.glob ('../practice_godgil/*.png')+glob.glob('../practice_godgil/*.jpg')
    imgs_path = '/mnt/hdd2/bdd100k/bdd100k/images/100k/val/'
    imgs_ext = '*.jpg'

    imgs2_path = '/mnt/hdd2/JM/bdd100k/drivable_maps/color_labels/val/'
    imgs2_ext = '*.png'

    imgs = glob.glob(os.path.join(imgs_path, imgs_ext))  # +('../label/*.png')
    imgs2 = glob.glob(os.path.join(imgs2_path, imgs2_ext))
    imgs = sorted(imgs)  # real image
    imgs2 = sorted(imgs2)  # color label
    # np.random.shuffle(imgs)
    threshold = 0.8

    print(len(imgs))
    # for path in imgs[:-1]:

    mean_iou = []
    cnt=0;
    for path, path2 in zip(imgs, imgs2):
        img = cv.imread(path)
        img = cv.resize(img, (512, 288), interpolation=cv.INTER_CUBIC)

        img2 = cv.imread(path2)

        img2 = cv.resize(img2, (512, 288), interpolation=cv.INTER_CUBIC)

        predict = np.reshape(model.predict(np.expand_dims(img, axis=0) / 255), [288, 512, 3]) * 255

        predict[predict < (255 * threshold)] = 0
        predict[:, :, 1] = 0

        imgpred = cv.add(img, predict, dtype=cv.CV_8U)
        imgpred2 = cv.add(img2, predict, dtype=cv.CV_8U)
        # cv.imshow('imgr',predict)
        # cv.waitKey()

        # cv.imwrite('result_image/godgil/{}'.format(path.split('/')[-1]), imgpred)  # [-1] is last split
        # cv.imwrite('result_image/godgil/{}'.format(path2.split('/')[-1]), predict)

        y_pred = np.array(predict)
        y_pred = np.expand_dims(y_pred, axis=0) / 255
        y_pred = y_pred.astype('float32')

        y_true = np.array(img2)
        y_true = np.expand_dims(img2, axis=0) / 255
        y_true = y_true.astype('float32')
        # cv.imshow('imgr2',img2)
        # cv.waitKey()
        # cv.destroyAllWindows()
        iou_acc = Model.iou_acc(y_true, y_pred)
        #iou_acc = tf.print(iou_acc)
        mean_iou.append(iou_acc)
        cnt+=1
        print(cnt)
        print(path.split('/')[-1], "\tiou_acc :{} ".format(iou_acc))

    mean=tf.print(sum(mean_iou, 0.0) / len(mean_iou))
    print("Mean IoU : ", sum(mean_iou, 0.0) / len(mean_iou))
    # cv.imshow('imgpred', imgpred)
    # cv.imshow('original', img)
    # cv.waitKey()
