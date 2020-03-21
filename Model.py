import glob
import os

from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras.optimizers import Adam
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

    def up(self, tensor1, tensor2, ch):
        with K.name_scope('up'):
            x = self.conv_bn_relu(L.UpSampling2D()(tensor1), ch)
            x = self.concat(x, tensor2)
            return x

    @staticmethod
    def conv_bn_relu(tensor, ch, atrous=(1, 1)):
        with K.name_scope('3X3_conv_bn_relu'):
            x = L.Conv2D(ch, 3, padding='same', dilation_rate=atrous)(tensor)
            x = L.BatchNormalization()(x)
            x = L.ReLU()(x)
            return x

    def gru_relu(self, tensor, direction):
        with K.name_scope('gru_relu'):
            h, w, c = K.shape(tensor)[1:]
            if direction == 'h':
                tensor = K.reshape(tensor, [-1, w, h * c])
                a = L.GRU(tensor, activation='relu', return_sequences=True, go_backwards=False)
                b = L.GRU(tensor, activation='relu', return_sequences=True, go_backwards=True)
                x = self.add(a, b)
                x = K.reshape(x, [-1, h, w, c])
                return x
            if direction == 'v':
                tensor = K.reshape(tensor, [-1, h, w * c])
                a = L.GRU(tensor, activation='relu', return_sequences=True, go_backwards=False)
                b = L.GRU(tensor, activation='relu', return_sequences=True, go_backwards=True)
                x = self.add(a, b)
                x = K.reshape(x, [-1, h, w, c])
                return x

    def layer_rnn(self, tensor):
        with K.name_scope('layer_rnn'):
            x = self.gru_relu(tensor, 'h')
            x = self.gru_relu(x, 'v')
            return x

    def resnet(self, tensor, ch):
        with K.name_scope('resnet'):
            x = self.conv_bn_relu(tensor, ch)
            y = self.conv_bn_relu(x, ch)
            y = self.conv_bn_relu(y, ch)
            x = self.add(x, y)
            return x

    def atrous(self, tensor, ch):
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
        tensor = Input([720, 1280, 3])

        d1 = self.conv_bn_relu(tensor, self.channel)
        d1 = self.layer_rnn(d1)

        d2 = self.conv_bn_relu(d1, self.channel)
        d2 = self.layer_rnn(d2)

        e = self.last_conv(d2)

        self.model = Model(tensor, e)
        self.model.compile('adam', iou_loss, [iou_acc])
        self.model.summary()

    def load(self, answer=None):
        models_path = glob.glob('D:/Models/*.h5')
        if len(models_path):
            latest = max(models_path, key=os.path.getctime).replace('\\', '/')
            print('Loaded ' + str(latest))
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
                elif len(ans.replace('\n', '')) == 0 or ans == 'y':
                    self.model = models.load_model(filepath, {'iou_loss': iou_loss, 'iou_acc': iou_acc})
                    self.epoch = filepath.split('-')[0].split('_')[-1]
                    print('Loaded Model', self.epoch)
                else:
                    print('\033[35m' + ans + '\033[0m', 'is invalid. Please type again.')
                    continue
        else:
            self.build()
