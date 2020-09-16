from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import tensorflow as tf
import Data
import Model
# import myslack
import os
import argparse
from tensorflow.python.client import device_lib
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--gpus', default='3', type=str, help='Which GPUs you want to use? (0,1,2,3)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
# myslack.send_slack("start")

# path = 'D:/Models/'
path = 'Models/gpu2/'

# path = 'Models/'
#gpus = tf.config.experimental.list_logical_devices('GPUS')
#if gpus:
#   tf.config.experimental.set_memory_growth(gpus[0], True)


def scheduler(epoch):
    warmup = 3
    warmup_lr = 1e-5  # 0.00001
    threshold = 15
    lr = 1e-4  # 0.0001
    lr2 = 5e-5  # 0.00005
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
    # TensorBoard('./logs/', profile_batch=2)
]
#with tf.device('/XLA_GPU:0'):
b = 4
tr_batch = Data.Load_tr(batch_size=b)
te_batch = Data.Load_te(batch_size=b)
print(tr_batch)
c = 3
model = Model.SegModel(3)
model.load()
model.fit(tr_batch, te_batch, callback)
# myslack.send_slack("finish")
