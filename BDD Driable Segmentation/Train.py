from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import Load
import Model

path = 'D:/Saved Model/'

callback = [
    ReduceLROnPlateau(factor=0.2, patience=6, verbose=1),
    ModelCheckpoint(path + 'model_{val_acc:.4f}.h5', verbose=1, save_best_only=True)
]

tr_batch, te_batch = Load.LoadData()

model = Model.Build()
