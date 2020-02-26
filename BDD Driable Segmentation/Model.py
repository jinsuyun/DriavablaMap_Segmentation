from tensorflow.keras import layers as L
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
import glob
import os


def Loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred, True, 0.1)


def Conv_block(input, filter, kernel=3):
    x = L.Conv2D(filter, kernel, padding='same')(input)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x


def Upsampling_block(input1, input2):
    y = Conv_block(input2, input1.shape[-1], 1)
    x = L.UpSampling2D()(input1)
    x = L.Add()([x, y])
    return x


def LoadSavedModel():
    models = glob.glob('D:/Model/*.h5')
    if len(models):
        latest = max(models, key=os.path.getctime).replace('\\', '/')
        print('Loaded ' + str(latest))
        return 1, latest
    else:
        print('Model Not Founded.')
        return 0, None


def Build():
    tensor = Input([180, 320, 3])
    ch = 8

    d1 = Conv_block(tensor, ch)
    d1 = Conv_block(d1, ch)
    m = L.MaxPool2D()(d1)

    d2 = Conv_block(m, ch * 2)
    d2 = Conv_block(d2, ch * 2)
    m = L.MaxPool2D()(d2)

    e = Conv_block(m, ch * 4)
    e = Conv_block(e, ch * 4)
    m = Upsampling_block(e, d2)

    u2 = Conv_block(m, ch * 2)
    u2 = Conv_block(u2, ch * 2)
    m = Upsampling_block(u2, d1)

    u1 = Conv_block(m, ch)
    u1 = Conv_block(u1, ch)

    e = Conv_block(u1, 3)

    model = Model(tensor, e)
    model.summary()
    model.compile(Adam(epsilon=0.001), Loss, ['acc'])
    exist, filepath = LoadSavedModel()
    if exist:
        model.load_weights(filepath)
    return model
