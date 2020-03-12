from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import tensorflow as tf
import Data
import Model

path = 'D:/Models/'
gpus = tf.config.experimental.list_logical_devices('GPUS')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

lr = 3e-4


def scheduler(epoch):
    warmup = 5
    warmup_lr = 1e-5
    threshold = 30
    lr2 = 1e-4
    if epoch < warmup:
        return warmup_lr
    elif epoch < threshold:
        return lr
    else:
        return lr2


callback = [
    ModelCheckpoint(path + 'model_{epoch:02d}-{val_iou_acc:.4f}_{iou_acc:.4f}.h5'),
    LearningRateScheduler(scheduler, verbose=1),
    TensorBoard('D:/logs/', profile_batch=10000)
]

tr_batch, te_batch = Data.Load()

model, epoch = Model.LoadSavedModel(lr)
model.fit(tr_batch, epochs=epoch + 1, verbose=1, callbacks=callback, validation_data=te_batch, initial_epoch=epoch)
