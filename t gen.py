from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers as L
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np

(trimg, trlabel), (teimg, telabel) = cifar10.load_data()


class Generator(Sequence):
    def __init__(self, img, label, batch):
        self.img, self.label = img, label
        self.batch = batch

    def __len__(self):
        return np.int(np.floor(len(self.img) / (len(self.img) / float(self.batch))))

    def __getitem__(self, idx):
        img_batch = self.img[idx * self.batch:(idx + 1) * self.batch] / 255.
        label_batch = to_categorical(self.label[idx * self.batch:(idx + 1) * self.batch], 10)
        return img_batch, label_batch

    def on_epoch_end(self):
        pass


def Loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred, True)


def Conv_block(input, filter):
    x = L.Conv2D(filter, 3, padding='same')(input)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x


tensor = Input([32, 32, 3])

t = Conv_block(tensor, 32)
t = L.MaxPool2D()(t)
t = Conv_block(t, 64)
t = L.MaxPool2D()(t)
t = Conv_block(t, 128)

t = L.GlobalAvgPool2D()(t)
t = L.Dense(32, 'relu')(t)
t = L.Dense(10, 'softmax')(t)

batch = 512
model = Model(tensor, t)
model.compile(Adam(epsilon=1e-3), Loss, ['acc'])
model.fit(Generator(trimg, trlabel, batch), epochs=70,
          validation_data=Generator(teimg, telabel, batch // 4), verbose=2)
