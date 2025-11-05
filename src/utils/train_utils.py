"""
Утилиты для обучения и сохранения модели.
"""
from keras.callbacks import ModelCheckpoint


def train_model(model, enc_inputs, dec_inputs, dec_targets, epochs=30, batch_size=64, name='model'):
    ckpt = ModelCheckpoint(f'{name}.h5', monitor='val_loss', save_best_only=True)
    history = model.fit(
        [enc_inputs, dec_inputs], dec_targets,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt]
    )
    return history
