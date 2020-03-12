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


def BN_ReLU(input):
    x = L.BatchNormalization()(input)
    x = L.ReLU()(x)
    return x


def Softmax(input):
    x = L.Softmax()(input)
    return x


def Conv_layer(input, filter, kernel=3):
    x = L.Conv2D(filter, kernel, padding='same')(input)
    return x


def Trans_layer(input1, input2):
    x = Conv_layer(L.UpSampling2D()(input1), input2.shape[-1])
    x = BN_ReLU(x)
    x = L.Concatenate()([x, input2])
    return x


def Pool_layer(input):
    x = L.MaxPool2D()(input)
    return x


def Build(lr=1e-3):
    tensor = Input([288, 512, 3])
    ch = 32

    d1 = Conv_layer(tensor, ch)
    d1 = BN_ReLU(d1)
    d1 = Conv_layer(d1, ch)
    d1 = BN_ReLU(d1)

    m = Pool_layer(d1)
    d2 = Conv_layer(m, ch * 2)
    d2 = BN_ReLU(d2)

    m = Pool_layer(d2)
    d3 = Conv_layer(m, ch * 4)
    d3 = BN_ReLU(d3)

    m = Pool_layer(d3)
    d4 = Conv_layer(m, ch * 6)
    d4 = BN_ReLU(d4)

    m = Pool_layer(d4)
    d5 = Conv_layer(m, ch * 8)
    d5 = BN_ReLU(d5)

    m = Pool_layer(d5)
    d6 = Conv_layer(m, ch * 10)
    d6 = BN_ReLU(d6)
    m = Trans_layer(d6, d5)

    d5 = Conv_layer(m, ch * 8)
    d5 = BN_ReLU(d5)
    m = Trans_layer(d5, d4)

    d4 = Conv_layer(m, ch * 6)
    d4 = BN_ReLU(d4)
    m = Trans_layer(d4, d3)

    d3 = Conv_layer(m, ch * 4)
    d3 = BN_ReLU(d3)
    m = Trans_layer(d3, d2)

    d2 = Conv_layer(m, ch * 2)
    d2 = BN_ReLU(d2)
    m = Trans_layer(d2, d1)

    d1 = Conv_layer(m, ch)
    d1 = BN_ReLU(d1)
    d1 = Conv_layer(d1, ch)
    d1 = BN_ReLU(d1)

    e = Conv_layer(d1, 3, kernel=1)
    e = Softmax(e)

    model = Model(tensor, e)
    model.compile(Adam(learning_rate=lr), iou_loss, [iou_acc])
    model.summary()
    return model


def LoadSavedModel(lr=1e-3, answer=None):
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
                model = Build(lr)
                print('Passed')
                return model, 0
            elif len(ans.replace('\n', '')) == 0 or ans == 'y':
                model = models.load_model(filepath, {'iou_loss': iou_loss, 'iou_acc': iou_acc})
                epoch = filepath.split('-')[0].split('_')[-1]
                print('Loaded Model', epoch)
                return model, int(epoch)
            else:
                print('\033[35m' + ans + '\033[0m', 'is invalid. Please type again.')
                continue
    else:
        model = Build(lr)
        return model, 0
