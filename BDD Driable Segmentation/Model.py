from tensorflow.keras import layers as L
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam


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


def Build():
    tensor = Input([320, 180, 3])

    d1 = Conv_block(tensor, 16)
    d1 = Conv_block(d1, 16)
    m = L.MaxPool2D()(d1)

    d2 = Conv_block(m, 32)
    d2 = Conv_block(d2, 32)
    m = L.MaxPool2D()(d2)

    e = Conv_block(m, 64)
    e = Conv_block(e, 64)
    m = Upsampling_block(e, d2)

    u2 = Conv_block(m, 32)
    u2 = Conv_block(u2, 32)
    m = Upsampling_block(u2, d1)

    u1 = Conv_block(m, 16)
    u1 = Conv_block(u1, 16)

    e = Conv_block(u1, 3)

    model = Model(tensor, e)
    model.summary()
    model.compile(Adam(epsilon=0.001), Loss, ['acc'])
    return model
