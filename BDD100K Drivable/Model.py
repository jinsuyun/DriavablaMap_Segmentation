from tensorflow.keras import Input, Model
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
import tensorflow as tf
import glob
import os
import Generator
import Load


def Conv(tensor, ch, kernel=3):
    x = L.Conv2D(ch, kernel, padding='same')(tensor)
    x = L.BatchNormalization()(x)
    x = Mish(x)
    return x


def Mish(tensor):
    return tensor * (K.tanh(K.softplus(tensor)))


def Main(train=True):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    if train:
        (trimg, trlabel), (teimg, telabel) = Load.Main()
        trimg, trlabel = Generator.Main(trimg, trlabel)
    else:
        trimg = tf.zeros([1, 180, 320, 3])

    _, height, width, channel = trimg.shape
    tensor = Input([height, width, channel])

    ch = 16
    # 180, 320
    s1 = Conv(tensor, ch)
    s1 = Conv(s1, ch)
    r1 = Conv(s1, ch * 2, 1)

    # 90, 160
    s1_ = L.MaxPool2D()(s1)
    s2 = Conv(s1_, ch * 2)
    s2 = Conv(s2, ch * 2)
    r2 = Conv(s2, ch * 4, 1)

    # 45, 80
    s2_ = L.MaxPool2D()(s2)
    lt = Conv(s2_, ch * 4)
    lt = Conv(lt, ch * 4)
    u2_ = L.UpSampling2D()(lt)

    # 90, 160
    r2 = L.Add()([u2_, r2])
    u2 = Conv(r2, ch * 2)
    u2 = Conv(u2, ch * 2)
    u1_ = L.UpSampling2D()(u2)

    # 180, 320
    r1 = L.Add()([u1_, r1])
    u1 = Conv(r1, ch)
    u1 = Conv(u1, ch)

    out = Conv(u1, 3)

    model = Model(tensor, out)
    model.compile('adam', 'categorical_crossentropy', ['acc'])
    model.summary()
    model.load_weights(LoadSavedModel())
    if train:
        model.fit(trimg, trlabel, 4, 20, verbose=2, callbacks=Callback_list(), validation_data=(teimg, telabel))
    else:
        return model


def LoadSavedModel():
    models = glob.glob('D:/Model/*.h5')
    if len(models):
        latest = max(models, key=os.path.getctime).replace('\\', '/')
        print('Loaded ' + str(latest))
        return latest
    else:
        print('Model Not Founded.')


def Callback_list():
    lists = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.25, patience=8, verbose=1),
        ModelCheckpoint('D:/Model/model_{val_acc:.4f}.h5', verbose=1)
    ]
    return lists
