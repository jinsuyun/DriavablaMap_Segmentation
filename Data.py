from tensorflow.keras.utils import Sequence
import cv2 as cv
import numpy as np
import glob

# ds_path = 'C:/bdd100k/'
# ds_path = 'bdd100k/'
# ds_path='../bdd100k/bdd100k/images/100k/'
ds_path = '/mnt/hdd2/bdd100k/bdd100k/images/100k/'
label_path = '/mnt/hdd2/JM/bdd100k/'
cnt = 0


def Load_tr(batch_size=4):
    trlabel = glob.glob(label_path + 'drivable_maps/color_labels/train/*.png')
    ##trlabel = glob.glob(ds_path + 'drivable_maps/color_labels/train/*.png')
    tr_batch = Generator('train', trlabel, batch_size, 20)
    return tr_batch


def Load_te(batch_size=4):
    telabel = glob.glob(label_path + 'drivable_maps/color_labels/val/*.png')
    ##telabel = glob.glob(ds_path + 'drivable_maps/color_labels/val/*.png')
    te_batch = Generator('val', telabel, batch_size, 30)
    return te_batch


class Generator(Sequence):

    def __init__(self, place, label, batch, div=1):
        self.place, self.label, self.batch, self.div = place, label, batch, div
        np.random.shuffle(self.label)
        self.match(self.label, self.place)

    def __len__(self):
        return int(np.floor(len(self.label) / self.div / float(self.batch)))

    def __getitem__(self, item):
        self.get_batch(self.img[item * self.batch:(item + 1) * self.batch],
                       self.label[item * self.batch:(item + 1) * self.batch])
        return np.asarray(self.img_batch), np.asarray(self.label_batch)

    def get_batch(self, img, label):
        global cnt
        self.img_batch, self.label_batch = [], []
        for img, label in zip(img, label):
            img = cv.imread(img)
            label = cv.imread(label)

            ##cv.imshow("",img)
            img = img / 255
            label = label / 255
            # cv.imwrite('result_image/godgil/imgimg{}.jpg'.format(cnt), img)
            # cv.imwrite('result_image/godgil/labellabel{}.jpg'.format(cnt), label)

            label[(label[:, :, 0] == 0) & (label[:, :, 2] == 0)] = [[[0, 1, 0]]]

            # img = cv.resize(img, (512, 288), interpolation=cv.INTER_CUBIC)
            # label = cv.resize(label, (512, 288), interpolation=cv.INTER_NEAREST)
            img = cv.resize(img, (512, 288), interpolation=cv.INTER_LINEAR)
            # blur 추�?부�?            img = cv.GaussianBlur(img, (3, 3), 0)
            label = cv.resize(label, (512, 288), interpolation=cv.INTER_LINEAR)

            cnt += 1
            self.img_batch.append(img)
            self.label_batch.append(label)

    def match(self, dirs, place):
        self.img = []
        for path in dirs:
            identity = path.split('/')[-1].split('_')[0] + '.jpg'
            ds_path = '/mnt/hdd2/bdd100k/bdd100k/images/100k/'
            path = ds_path + place + '/' + identity
            ##path = ds_path + 'images/100k/' + place + '/' + identity
            self.img.append(path)

    def on_epoch_end(self):
        np.random.shuffle(self.label)
        self.match(self.label, self.place)


if __name__ == '__main__':
    Load_tr()
