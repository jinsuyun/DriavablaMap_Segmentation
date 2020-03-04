import glob
import os

from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models


def iou_acc(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss(y_true, y_pred, smooth=1):
    return 1 - iou_acc(y_true, y_pred, smooth)


def Conv_block(input, filter, kernel=3, last=False):
    x = L.Conv2D(filter, kernel, padding='same')(input)
    x = L.BatchNormalization()(x)
    if last:
        x = L.Softmax()(x)
    else:
        x = L.ReLU()(x)
    return x


def Upsampling_block(input1, input2):
    y = Conv_block(input2, input1.shape[-1], kernel=1)
    x = L.UpSampling2D()(input1)
    x = L.Add()([x, y])
    return x


def Build():
    tensor = Input([288, 512, 3])
    ch = 32

    d1 = Conv_block(tensor, ch)

    m = L.MaxPool2D()(d1)
    d2 = Conv_block(m, ch * 2)

    m = L.MaxPool2D()(d2)
    d3 = Conv_block(m, ch * 4)

    m = L.MaxPool2D()(d3)
    d4 = Conv_block(m, ch * 6)

    m = L.MaxPool2D()(d4)
    d5 = Conv_block(m, ch * 8)

    m = L.MaxPool2D()(d5)
    e = Conv_block(m, ch * 10)
    e = Conv_block(e, ch * 10)
    m = Upsampling_block(e, d5)

    u5 = Conv_block(m, ch * 8)
    m = Upsampling_block(u5, d4)

    u4 = Conv_block(m, ch * 6)
    m = Upsampling_block(u4, d3)

    u3 = Conv_block(m, ch * 4)
    m = Upsampling_block(u3, d2)

    u2 = Conv_block(m, ch * 2)
    m = Upsampling_block(u2, d1)

    u1 = Conv_block(m, ch)

    e = Conv_block(u1, 3, kernel=1, last=True)

    model = Model(tensor, e)
    model.compile(Adam(epsilon=1e-5), iou_loss, [iou_acc])
    model.summary()
    return model


def LoadSavedModel():
    models_path = glob.glob('./Models/*.h5')
    if len(models_path):
        latest = max(models_path, key=os.path.getctime).replace('\\', '/')
        print('Loaded ' + str(latest))
        exist, filepath = 1, latest
    else:
        print('Model Not Founded.')
        exist, filepath = 0, None

    if exist:
        ans = input('Load? ([y]/n)')
        if ans == 'n':
            model = Build()
            print('Passed')
            return model
        else:
            model = models.load_model(filepath, {'iou_loss': iou_loss, 'iou_acc': iou_acc})
            print('Loaded Model')
            return model
    else:
        model = Build()
        return model
