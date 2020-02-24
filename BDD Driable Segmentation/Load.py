from tensorflow.keras.utils import Sequence
import cv2 as cv
import numpy as np
import glob

ds_path = 'D:/bdd100k/'


def Match_(dirs, place):
    array = []
    for path in dirs:
        identity = path.split('\\')[-1].split('_')[0] + '.jpg'
        path = ds_path + 'images/100k/' + place + '/' + identity
        array.append(path)
    return array


def LoadData(shuffle=True, batch_size=17500):
    trlabel = glob.glob(ds_path + 'drivable_maps/labels/train/*.png')
    telabel = glob.glob(ds_path + 'drivable_maps/labels/val/*.png')

    if shuffle:
        np.random.shuffle(trlabel)
        np.random.shuffle(telabel)

    trimg = Match_(trlabel, 'train')
    teimg = Match_(telabel, 'val')

    tr_batch = BatchGenerator(trimg, trlabel, batch_size)
    te_batch = BatchGenerator(teimg, telabel, batch_size)

    return tr_batch, te_batch


class BatchGenerator(Sequence):
    def __init__(self, img, label, batch):
        self.img, self.label, self.batch = img, label, batch

    def __len__(self):
        return int(np.floor(len(self.img) / float(self.batch)))

    @staticmethod
    def LoadImg_(dirs, div):
        array = []
        for dir in dirs:
            img = cv.imread(dir, cv.IMREAD_REDUCED_COLOR_4)
            if div:
                cv.imshow('loded', img)
                cv.waitKey(1)
                img = img / 255
            array.append(img)
        return np.array(array)

    def __getitem__(self, index):
        img_batch = self.img[index * self.batch: (index + 1) * self.batch]
        label_batch = self.label[index * self.batch: (index + 1) * self.batch]
        img_batch = self.LoadImg_(img_batch, True)
        label_batch = self.LoadImg_(label_batch, False)
        return img_batch, label_batch
