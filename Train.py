from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import tensorflow as tf
import Data
import Model

path = 'D:/Models/'
gpus = tf.config.experimental.list_logical_devices('GPUS')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

lr = 2e-4


def scheduler(epoch):
    warmup = 5
    warmup_lr = 1e-5
    threshold = 25
    lr2 = 5e-5
    if epoch < warmup:
        return warmup_lr
    elif epoch == warmup:
        return (lr + warmup_lr) / 2
    elif epoch < threshold:
        return lr
    else:
        return lr2


callback = [
    ModelCheckpoint(path + 'model_{epoch:02d}-{val_iou_acc:.4f}_{iou_acc:.4f}.h5'),
    LearningRateScheduler(scheduler, verbose=1),
    TensorBoard('D:/logs/', profile_batch=10000)
]

s = 4
tr_batch = Data.Load_tr(batch_size=s)
te_batch = Data.Load_te(batch_size=s)

model, epoch = Model.LoadSavedModel(lr)
model.fit(tr_batch, epochs=epoch + 1, verbose=1, callbacks=callback, validation_data=te_batch, initial_epoch=epoch)
