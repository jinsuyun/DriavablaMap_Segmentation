import glob
import os

from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras import models


def iou_acc(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss(y_true, y_pred, smooth=1):
    return 1 - iou_acc(y_true, y_pred, smooth)


class SegModel:
    def __init__(self, class_num, channel=32):
        self.channel = channel
        self.class_num = class_num
        self.epoch = 0

    @staticmethod
    def add(a, b):
        with K.name_scope('add'):
            x = L.Add()([a, b])
            return x

    @staticmethod
    def concat(a, b):
        with K.name_scope('concat'):
            x = L.Concatenate()([a, b])
            return x

    @staticmethod
    def down(tensor):
        with K.name_scope('down'):
            x = L.MaxPool2D()(tensor)
            return x

    def up(self, tensor1, tensor2):
        with K.name_scope('up'):
            x = self.conv_bn_relu(L.UpSampling2D()(tensor1), tensor2.shape.as_list()[-1])
            x = self.add(x, tensor2)
            return x

    @staticmethod
    def conv_bn_relu(tensor, ch, atrous=(1, 1)):
        with K.name_scope('3X3_conv_bn_relu'):
            x = L.Conv2D(ch, 3, padding='same', dilation_rate=atrous)(tensor)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
            return x

    def resnet(self, tensor, ch):
        with K.name_scope('resnet'):
            x = self.conv_bn_relu(tensor, ch)
            y = self.conv_bn_relu(x, ch)
            y = self.conv_bn_relu(y, ch)
            x = self.add(x, y)
            return x

    def deeplab(self, tensor, ch):
        with K.name_scope('atrous'):
            a = self.conv_bn_relu(tensor, ch, (1, 1))
            b = self.conv_bn_relu(tensor, ch, (3, 3))
            x = self.add(a, b)
            return x

    def last_conv(self, tensor):
        with K.name_scope('last_conv'):
            x = L.Conv2D(self.class_num, 1, padding='same')(tensor)
            x = L.Softmax()(x)
            return x

    def build(self):
        tensor = Input([288, 512, 3])

        d1 = self.conv_bn_relu(tensor, self.channel)
        d1 = self.conv_bn_relu(d1, self.channel)

        m = self.down(d1)
        d2 = self.conv_bn_relu(m, self.channel * 2)

        m = self.down(d2)
        d3 = self.conv_bn_relu(m, self.channel * 4)

        m = self.down(d3)
        d4 = self.conv_bn_relu(m, self.channel * 6)

        m = self.down(d4)
        d5 = self.conv_bn_relu(m, self.channel * 8)

        m = self.down(d5)
        d6 = self.conv_bn_relu(m, self.channel * 10)
        m = self.up(d6, d5)

        d5 = self.conv_bn_relu(m, self.channel * 8)
        m = self.up(d5, d4)

        d4 = self.conv_bn_relu(m, self.channel * 6)
        m = self.up(d4, d3)

        d3 = self.conv_bn_relu(m, self.channel * 4)
        m = self.up(d3, d2)

        d2 = self.conv_bn_relu(m, self.channel * 2)
        m = self.up(d2, d1)

        d1 = self.conv_bn_relu(m, self.channel)
        d1 = self.conv_bn_relu(d1, self.channel)

        e = self.last_conv(d1)

        self.model = Model(tensor, e)
        self.model.compile('adam', iou_loss, [iou_acc])
        self.model.summary()

    def fit(self, tr_batch, te_batch, callback):
        self.model.fit(tr_batch, validation_data=te_batch, callbacks=callback,
                       epochs=self.epoch + 1, initial_epoch=self.epoch)

    def predict(self, img):
        return self.model.predict(img)

    def load(self, answer=None):
        models_path = glob.glob('D:/Models/*.h5')
        if len(models_path):
            latest = max(models_path, key=os.path.getctime).replace('\\', '/')
            print('Found ' + str(latest))
            exist, filepath = 1, latest
        else:
            print('Model Not Founded.')
            exist, filepath = 0, None

        stop = False
        if exist:
            while True:
                if answer is not None and stop is False:
                    ans = answer
                    stop = True
                else:
                    ans = input('Load? ([y]/n)')
                if ans == 'n':
                    self.build()
                    print('Passed')
                    break
                elif len(ans.replace('\n', '')) == 0 or ans == 'y':
                    self.model = models.load_model(filepath, {'iou_loss': iou_loss, 'iou_acc': iou_acc})
                    self.epoch = int(filepath.split('-')[0].split('_')[-1])
                    print('Loaded Model', self.epoch)
                    break
                else:
                    print('\033[35m' + ans + '\033[0m', 'is invalid. Please type again.')
                    continue
        else:
            self.build()


if __name__ == '__main__':
    model = SegModel(3)
    model.load()
