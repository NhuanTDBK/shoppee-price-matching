import gc
import os
import random
import re

import tensorflow_addons as tfx

from modelling.callbacks import *


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def get_opt(params):
    if params["name"] == "adam":
        return tfx.optimizers.AdamW(learning_rate=params["lr"], weight_decay=params["wd"])

    elif params["name"] == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=params["lr"], momentum=params["momentum"], )


def get_tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))

    strategy = tf.distribute.TPUStrategy(resolver)

    return strategy


def get_gpu_strategy():
    mirrored_strategy = tf.distribute.MirroredStrategy()
    return mirrored_strategy


def count_data_items(filenames):
    # The number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def train(params: dict, model_fn,
          optimizer: tf.optimizers.Optimizer,
          loss: tf.keras.losses.Loss, metrics, callbacks, ds_train, ds_val=None, num_training_images=None,
          model_saved_dir=None, model_name=None):
    if params["use_tpu"]:
        strategy = get_tpu_strategy()
    else:
        strategy = get_gpu_strategy()

    with strategy.scope():
        model, emb_model = model_fn()
        model.compile(optimizer, loss, metrics)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(0))

    ckpt_dir = os.path.join(model_saved_dir, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=2, )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    if not callbacks:
        callbacks = []

    if not any([isinstance(cb, EarlyStoppingByLossVal) for cb in callbacks]):
        callbacks.append(EarlyStoppingByLossVal(monitor="sparse_categorical_accuracy", value=0.91), )
    if not any([isinstance(cb, tf.keras.callbacks.TensorBoard) for cb in callbacks]):
        callbacks.append(tf.keras.callbacks.TensorBoard(write_graph=False, histogram_freq=5, update_freq=5, ))
    if not any([isinstance(cb, tf.keras.callbacks.CSVLogger) for cb in callbacks]):
        callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(model_saved_dir, "training_%s.log" % model_name)), )

    if not any([isinstance(cb, CheckpointCallback) for cb in callbacks]):
        callbacks.append(CheckpointCallback(ckpt_manager))

    steps_per_epoch = num_training_images // params["batch_size"]
    epochs = params["epochs"]

    if ckpt_manager.latest_checkpoint:
        print("Restored from: ", ckpt_manager.latest_checkpoint)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        epochs -= tf.keras.backend.get_value(ckpt.epoch)
    else:
        print("Start from scratch")

    model.fit(ds_train,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=ds_val,
              callbacks=callbacks)

    emb_model.save_weights(os.path.join(model_saved_dir, "emb_" + str(model_name)), save_format="h5", overwrite=True)

    del model, emb_model
    gc.collect()
    # return model, emb_model, optimizer, loss, metrics
