from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import Data
import Model

path = 'D:/Model/'

callback = [
    ReduceLROnPlateau(factor=0.2, patience=6, verbose=1),
    ModelCheckpoint(path + 'model_{val_acc:.3f}_{acc:.2f}.h5', verbose=1)
]

tr_batch, te_batch = Data.Load()

model = Model.Build()
model.fit(tr_batch, epochs=10, verbose=2, callbacks=callback, validation_data=te_batch)
