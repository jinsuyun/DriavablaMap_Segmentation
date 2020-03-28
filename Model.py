import glob
import os

from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras import models

smooth = 1


def iou_acc(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss(y_true, y_pred):
    return 1 - iou_acc(y_true, y_pred)


class SegModel:
    def __init__(self, class_num, channel=32):
        self.channel = channel
        self.class_num = class_num
        self.epoch = 0

    @staticmethod
    def add(*args):
        with K.name_scope('add'):
            x = args[0]
            for y in args[1:]:
                x = L.Add()([x, y])
            return x

    @staticmethod
    def concat(*args):
        with K.name_scope('concat'):
            x = args[0]
            for y in args[1:]:
                x = L.Concatenate()([x, y])
            return x

    def down(self, tensor):
        with K.name_scope('down'):
            x = L.MaxPool2D([2, 2])(tensor)
            return x

    def up(self, tensor1, tensor2):
        with K.name_scope('up'):
            x = self.basic_conv(L.UpSampling2D()(tensor1), tensor2.shape.as_list()[-1])
            x = self.concat(x, tensor2)
            return x

    @staticmethod
    def sep_conv(tensor, ch, dilation=(1, 1), kernel=3):
        with K.name_scope('3X3_sep_conv_bn_relu'):
            x = L.DepthwiseConv2D(kernel, padding='same', dilation_rate=dilation)(tensor)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
            x = L.Conv2D(ch, 1, padding='valid')(x)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
            return x

    @staticmethod
    def basic_conv(tensor, ch, dilation=(1, 1), kernel=3):
        with K.name_scope('3X3_basic_conv_bn_relu'):
            x = L.Conv2D(ch, kernel, padding='same', dilation_rate=dilation)(tensor)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
            return x

    def point_conv(self, tensor):
        with K.name_scope('point_conv'):
            x = L.Conv2D(self.class_num, 1, padding='valid')(tensor)
            x = L.Softmax()(x)
            return x

    def build(self):
        tensor = Input([288, 512, 3])

        d1 = self.basic_conv(tensor, self.channel)
        d1 = self.sep_conv(d1, self.channel)

        m = self.down(d1)
        d2 = self.sep_conv(m, self.channel * 2)

        m = self.down(d2)
        d3 = self.sep_conv(m, self.channel * 4)

        m = self.down(d3)
        d4 = self.sep_conv(m, self.channel * 6)
        m = self.up(d4, d3)

        d3 = self.sep_conv(m, self.channel * 4)
        m = self.up(d3, d2)

        d2 = self.sep_conv(m, self.channel * 2)
        m = self.up(d2, d1)

        d1 = self.sep_conv(m, self.channel)
        d1 = self.basic_conv(d1, self.channel)

        e = self.point_conv(d1)

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
    model.load('n')
