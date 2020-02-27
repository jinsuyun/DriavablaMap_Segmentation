from tensorflow.keras.utils import Sequence
import cv2 as cv
import numpy as np
import glob

ds_path = 'C:/bdd100k/'


def Load(shuffle=True, batch_size=16):
    trlabel = glob.glob(ds_path + 'drivable_maps/labels/train/*.png')
    telabel = glob.glob(ds_path + 'drivable_maps/labels/val/*.png')

    if shuffle:
        np.random.shuffle(trlabel)
        np.random.shuffle(telabel)

    tr_batch = BatchGenerator_('train', trlabel, batch_size, True)
    te_batch = BatchGenerator_('val', telabel, batch_size, False)

    return tr_batch, te_batch


class BatchGenerator_(Sequence):
    @staticmethod
    def Match_(dirs, place):
        array = []
        for path in dirs:
            identity = path.split('\\')[-1].split('_')[0] + '.jpg'
            path = ds_path + 'images/100k/' + place + '/' + identity
            array.append(path)
        print(dirs[:2], array[:2])
        return array

    @staticmethod
    def Augmentation_(img, label, rotation=10, center=(320, 180)):
        width, height = center
        img = img / 255

        Rotate = np.random.rand(1) * rotation

        if np.random.rand(1) > 0.5:
            Rotate = -Rotate

        if np.random.rand(1) > 0.5:
            Flip = True
        else:
            Flip = False

        if Flip:
            img = cv.flip(img, 1)
            label = cv.flip(label, 1)

        M = cv.getRotationMatrix2D((height // 2, width // 2), Rotate, 1)
        img = cv.warpAffine(img, M, center)
        label = cv.warpAffine(label, M, center)
        return img, label

    def LoadImg_(self):
        img_array = []
        label_array = []
        for img_dir, label_dir in zip(self.img_batch, self.label_batch):
            img = cv.imread(img_dir, cv.IMREAD_REDUCED_COLOR_4)
            label = cv.imread(label_dir, cv.IMREAD_REDUCED_COLOR_4)

            if self.aug:
                img, label = self.Augmentation_(img, label, 10)

            img_array.append(img)
            label_array.append(label)
        return np.array(img_array), np.array(label_array)

    def __init__(self, place, label, batch, aug):
        self.place, self.label, self.batch, self.aug = place, label, batch, aug
        self.process = 0

    def __len__(self):
        return int(np.floor(len(self.label) / float(self.batch)))

    def __getitem__(self, index):
        self.process += 1
        if self.process % 100 == 0:
            print('Training..', self.process)
        self.img = self.Match_(self.label, self.place)
        self.img_batch = self.img[index * self.batch: (index + 1) * self.batch]
        self.label_batch = self.label[index * self.batch: (index + 1) * self.batch]
        img_batch, label_batch = self.LoadImg_()
        return img_batch, label_batch

    def on_epoch_end(self):
        np.random.shuffle(self.label)
