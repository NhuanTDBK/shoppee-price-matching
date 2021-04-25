import tensorflow as tf
import tensorflow_addons as tfx

def get_opt(params):
    if params["name"] == "adam":
        return tfx.optimizers.AdamW(learning_rate=params["lr"],weight_decay=params["wd"])

    elif params["name"] == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=params["lr"], momentum=params["momentum"],)