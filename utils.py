import os
import random
import gc

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


def train(params: dict, model: tf.keras.models.Model, emb_model: tf.keras.models.Model,
          optimizer: tf.optimizers.Optimizer,
          loss: tf.keras.losses.Loss, metrics, callbacks, ds_train, ds_val=None, num_training_images=None,
          model_saved_dir=None, model_name=None, checkpoint_dir='ckpt'):
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(0))
    ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(model_saved_dir, checkpoint_dir, model_name),
                                              max_to_keep=3, )

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
        callbacks.append(CheckpointCallback(ckpt_manager), )

    steps_per_epoch = num_training_images // params["batch_size"]

    if ckpt_manager.latest_checkpoint:
        print("Restored from: ", ckpt_manager.latest_checkpoint)
        ckpt.restore(ckpt_manager.latest_checkpoint)
    else:
        print("Start from scratch")

    # epochs = ckpt.epoch
    model.fit(ds_train,
              epochs=params["epochs"],
              steps_per_epoch=steps_per_epoch,
              validation_data=ds_val,
              callbacks=callbacks)

    emb_model.save_weights(os.path.join(model_saved_dir, "emb_" + str(model_name)), save_format="h5", overwrite=True)

    del model, emb_model
    gc.collect()
    # return model, emb_model, optimizer, loss, metrics
