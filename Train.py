from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import tensorflow as tf
import Data
import Model

path = 'D:/Models/'
gpus = tf.config.experimental.list_logical_devices('GPUS')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def scheduler(epoch):
    warmup = 3
    warmup_lr = 1e-5
    threshold = 25
    lr = 1e-4
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

b = 4
tr_batch = Data.Load_tr(batch_size=b)
te_batch = Data.Load_te(batch_size=b)

c = 3
model = Model.SegModel(3)
model.load()
model.fit(tr_batch, te_batch, callback)
