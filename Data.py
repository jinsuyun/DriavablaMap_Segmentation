from tensorflow.keras.utils import Sequence
import tensorflow as tf
import cv2 as cv
import numpy as np
import glob

ds_path = 'C:/bdd100k/'


def Load_tr(batch_size=4):
    trlabel = glob.glob(ds_path + 'drivable_maps/color_labels/train/*.png')
    np.random.shuffle(trlabel)
    tr_batch = BatchGenerator_('train', trlabel, batch_size, 20)
    return tr_batch


def Load_te(batch_size=4):
    telabel = glob.glob(ds_path + 'drivable_maps/color_labels/val/*.png')
    np.random.shuffle(telabel)
    te_batch = BatchGenerator_('val', telabel, batch_size, 30)
    return te_batch


class BatchGenerator_(Sequence):
    @staticmethod
    def Match_(dirs, place):
        array = []
        for path in dirs:
            identity = path.split('\\')[-1].split('_')[0] + '.jpg'
            path = ds_path + 'images/100k/' + place + '/' + identity
            array.append(path)
        return array

    def LoadImg_(self):
        img_array = []
        label_array = []
        for img_dir, label_dir in zip(self.img_batch, self.label_batch):
            cv.ocl.setUseOpenCL(True)
            img = cv.imread(img_dir)
            label = cv.imread(label_dir)

            # img = cv.GaussianBlur(img, (3, 3), 0)
            # noise = np.random.normal(0, 1, [288, 512, 3])
            # img = np.maximum(0, np.minimum(255, np.add(img, noise)))

            img = img / 255
            label = label / 255

            img = cv.resize(img, (512, 288), interpolation=cv.INTER_CUBIC)
            label = cv.resize(label, (512, 288), interpolation=cv.INTER_NEAREST)

            label[(label[:, :, 0] == 0) & (label[:, :, 2] == 0)] = [[[0, 1, 0]]]

            img_array.append(img)
            label_array.append(label)
        return np.array(img_array), np.array(label_array)

    def __init__(self, place, label, batch, div=1):
        self.place, self.label, self.batch, self.div = place, label, batch, div
        self.img = self.Match_(self.label, self.place)

    def __len__(self):
        return int(np.floor(len(self.label) / self.div / float(self.batch)))

    def __getitem__(self, index):
        self.img_batch = self.img[index * self.batch: (index + 1) * self.batch]
        self.label_batch = self.label[index * self.batch: (index + 1) * self.batch]
        img_batch, label_batch = self.LoadImg_()
        return img_batch, label_batch

    def on_epoch_end(self):
        # Tensorflow 버그로 작동 안함
        np.random.shuffle(self.label)
        self.Match_(self.label, self.place)


if __name__ == '__main__':
    Load_tr()
