import glob
import os

from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers as L
from tensorflow.keras.optimizers import Adam


def acc(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def loss(y_true, y_pred, smooth=1):
    return 1 - acc(y_true, y_pred, smooth)


def Conv_block(input, filter, kernel=3, last=False):
    x = L.Conv2D(filter, kernel, padding='same')(input)
    x = L.BatchNormalization()(x)
    if last:
        x = L.Softmax()(x)
    else:
        x = L.ReLU()(x)
    return x


def Upsampling_block(input1, input2):
    y = Conv_block(input2, input1.shape[-1], kernel=3)
    x = L.UpSampling2D()(input1)
    x = L.Add()([x, y])
    return x


def Build():
    tensor = Input([288, 512, 3])
    ch = 32

    d1 = Conv_block(tensor, ch)
    d1 = Conv_block(d1, ch)

    m = L.MaxPool2D()(d1)
    d2 = Conv_block(m, ch * 2)
    d2 = Conv_block(d2, ch * 2)

    m = L.MaxPool2D()(d2)
    d3 = Conv_block(m, ch * 3)
    d3 = Conv_block(d3, ch * 3)

    m = L.MaxPool2D()(d3)
    d4 = Conv_block(m, ch * 4)
    d4 = Conv_block(d4, ch * 4)

    m = L.MaxPool2D()(d4)
    d5 = Conv_block(m, ch * 5)
    d5 = Conv_block(d5, ch * 5)

    m = L.MaxPool2D()(d5)
    e = Conv_block(m, ch * 6)
    e = Conv_block(e, ch * 6)
    m = Upsampling_block(e, d5)

    u5 = Conv_block(m, ch * 5)
    u5 = Conv_block(u5, ch * 5)
    m = Upsampling_block(u5, d4)

    u4 = Conv_block(m, ch * 4)
    u4 = Conv_block(u4, ch * 4)
    m = Upsampling_block(u4, d3)

    u3 = Conv_block(m, ch * 3)
    u3 = Conv_block(u3, ch * 3)
    m = Upsampling_block(u3, d2)

    u2 = Conv_block(m, ch * 2)
    u2 = Conv_block(u2, ch * 2)
    m = Upsampling_block(u2, d1)

    u1 = Conv_block(m, ch)
    u1 = Conv_block(u1, ch)

    e = Conv_block(u1, 3, last=True)

    model = Model(tensor, e)
    model.compile(Adam(epsilon=1e-5), loss, [acc])
    model.summary()
    return model


def LoadSavedModel():
    models_path = glob.glob('D:/Model/*.h5')
    model = Build()
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
            print('Passed')
            return model
        else:
            model.load_weights(filepath)
            model.compile(Adam(epsilon=1e-5), loss, [acc])
            print('Loaded Model')
            return model
    else:
        return model
