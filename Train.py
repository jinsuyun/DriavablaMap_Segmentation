from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
import Data
import Model

path = 'D:/Models/'
gpus = tf.config.experimental.list_logical_devices('GPUS')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

lr = 1e-2


def scheduler(epoch):
    threshold = 10
    repeat = 4
    if epoch <= threshold:
        return lr
    else:
        diff = epoch - threshold
        return min(lr / (diff / repeat), lr)


callback = [
    ModelCheckpoint(path + 'model_{epoch:02d}-{val_iou_acc:.4f}_{iou_acc:.4f}.h5'),
    LearningRateScheduler(scheduler, verbose=1)
]

tr_batch, te_batch = Data.Load()

model, epoch = Model.LoadSavedModel(lr)
model.fit(tr_batch, epochs=epoch + 1, verbose=1, callbacks=callback, validation_data=te_batch, initial_epoch=epoch)
