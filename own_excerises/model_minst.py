import tensorflow as tf


tf.keras.backend.clear_session()
tf.random.set_seed(42)

def set_up_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[28,28]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200, activation="relu"))
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model

class PrintValTrainRatioCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                ratio = logs["val_loss"] / logs["loss"]
                print(f"Epoch={epoch}, val/train={ratio:.2f}")
