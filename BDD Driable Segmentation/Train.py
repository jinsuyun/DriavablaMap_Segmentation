from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import Data
import Model

path = 'D:/Model/'
gpus = tf.config.experimental.list_logical_devices('GPUS')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

callback = [
    ReduceLROnPlateau(factor=0.2, patience=5, verbose=1),
    ModelCheckpoint(path + 'model_{val_acc:.4f}_{acc:.4f}.h5')
]

tr_batch, te_batch = Data.Load()

model = Model.LoadSavedModel()
model.fit(tr_batch, epochs=20, verbose=1, callbacks=callback, validation_data=te_batch)
