import gc
import os
import random
import re

from modelling.callbacks import *


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


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


def get_disk_path():
    return "/content/drive/MyDrive/shopee-price"


def train(params: dict, model_fn,
          optimizer: tf.optimizers.Optimizer,
          loss: tf.keras.losses.Loss, metrics, callbacks, ds_train, ds_val=None, num_training_images=None,
          model_saved_dir=None, model_name=None):
    model, emb_model = model_fn()
    model.compile(optimizer, loss, metrics)

    path = os.path.join(model_saved_dir, "ckpt" + model_name, "fold")

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(0))

    ckpt_dir = os.path.join(model_saved_dir, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=1, )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    if not callbacks:
        callbacks = []

    if not any([isinstance(cb, EarlyStoppingByLossVal) for cb in callbacks]):
        callbacks.append(EarlyStoppingByLossVal(monitor="sparse_categorical_accuracy", value=0.91, verbose=1), )
    if not any([isinstance(cb, tf.keras.callbacks.CSVLogger) for cb in callbacks]):
        callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(model_saved_dir, "training_%s.log" % model_name)), )

    if not any([isinstance(cb, CheckpointCallback) for cb in callbacks]) and params["is_checkpoint"]:
        callbacks.append(CheckpointCallback(ckpt_manager, params["check_period"]))

    if not any([isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks]):
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(ckpt_dir, verbose=1, save_best_only=True, save_weights_only=True))

    if not any([isinstance(cb, tf.keras.callbacks.EarlyStopping) for cb in callbacks]):
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"],
                                                          restore_best_weights=True))

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

    print("Saved model to ", path)
    emb_model.save_weights(path, save_format="h5",
                           overwrite=True)

    del model, emb_model
    gc.collect()
    # return model, emb_model, optimizer, loss, metrics


def train_tpu(params: dict, model_fn,
              optimizer: tf.optimizers.Optimizer,
              callbacks, ds_train, ds_val=None, num_training_images=None,
              model_saved_dir=None, model_name=None, strategy: tf.distribute.TPUStrategy = None):
    ckpt_dir = os.path.join(model_saved_dir, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    emb_ckpt_path = os.path.join(model_saved_dir, "ckpt" + model_name, "fold")
    os.makedirs(emb_ckpt_path, exist_ok=True)

    with strategy.scope():
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy()

        model, emb_model = model_fn()
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(0))
        ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3, )
        epochs = params["epochs"]

        if ckpt_manager.latest_checkpoint:
            print("Restored from: ", ckpt_manager.latest_checkpoint)
            print(optimizer.iterations, optimizer.get_config())
            ckpt.restore(ckpt_manager.latest_checkpoint)
            current_epoch = tf.keras.backend.get_value(ckpt.epoch)
            epochs -= current_epoch
            print("Resume learning rate scheduler from {}".format(current_epoch))
            callbacks[0].count = current_epoch
        else:
            print("Start from scratch")

    if not callbacks:
        callbacks = []

    if not any([isinstance(cb, EarlyStoppingByLossVal) for cb in callbacks]):
        callbacks.append(EarlyStoppingByLossVal(monitor="sparse_categorical_accuracy", value=0.91, verbose=1), )
    if not any([isinstance(cb, tf.keras.callbacks.CSVLogger) for cb in callbacks]):
        callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(model_saved_dir, "training_%s.log" % model_name)), )

    if not any([isinstance(cb, CheckpointCallback) for cb in callbacks]) and params["is_checkpoint"]:
        callbacks.append(CheckpointCallback(ckpt_manager, params["check_period"]))

    if not any([isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks]):
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(emb_ckpt_path, verbose=1, save_best_only=True, save_weights_only=True))

    if not any([isinstance(cb, tf.keras.callbacks.EarlyStopping) for cb in callbacks]):
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"],
                                                          restore_best_weights=True))

    steps_per_epoch = num_training_images // params["batch_size"]

    model.fit(ds_train,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=ds_val,
              callbacks=callbacks)

    print("Saved model to ", emb_ckpt_path)
    emb_model.save_weights(emb_ckpt_path,
                           save_format="tf",
                           overwrite=True)

    del model, emb_model
    gc.collect()
    # return model, emb_model, optimizer, loss, metrics


def train_tpu_finetune(params: dict, model_fn,
                       optimizer: tf.optimizers.Optimizer,
                       callbacks, ds_train, ds_val=None, num_training_images=None,
                       model_saved_dir=None, model_name=None, strategy: tf.distribute.TPUStrategy = None):
    ckpt_dir = os.path.join(model_saved_dir, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    emb_ckpt_path = os.path.join(model_saved_dir, "ckpt" + model_name, "fold")
    os.makedirs(emb_ckpt_path, exist_ok=True)

    with strategy.scope():
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy()

        model, emb_model = model_fn()
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=tf.Variable(0))
        ckpt_manager = tf.train.CheckpointManager(ckpt, params["pretrained_path"], max_to_keep=3, )
        if ckpt_manager.latest_checkpoint:
            print("Restored from: ", ckpt_manager.latest_checkpoint)
            print(optimizer.iterations, optimizer.get_config())
            ckpt.restore(ckpt_manager.latest_checkpoint)

        print("Frozen batch norm")
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    if not callbacks:
        callbacks = []

    if not any([isinstance(cb, EarlyStoppingByLossVal) for cb in callbacks]):
        callbacks.append(EarlyStoppingByLossVal(monitor="sparse_categorical_accuracy", value=0.91, verbose=1), )
    if not any([isinstance(cb, tf.keras.callbacks.CSVLogger) for cb in callbacks]):
        callbacks.append(tf.keras.callbacks.CSVLogger(os.path.join(model_saved_dir, "training_%s.log" % model_name)), )

    if not any([isinstance(cb, CheckpointCallback) for cb in callbacks]) and params["is_checkpoint"]:
        ckpt_manager_new = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3, )
        callbacks.append(CheckpointCallback(ckpt_manager_new, params["check_period"]))

    if not any([isinstance(cb, tf.keras.callbacks.ModelCheckpoint) for cb in callbacks]):
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(emb_ckpt_path, verbose=1, save_best_only=True, save_weights_only=True))

    if not any([isinstance(cb, tf.keras.callbacks.EarlyStopping) for cb in callbacks]):
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params["patience"],
                                                          restore_best_weights=True))

    model.fit(ds_train,
              epochs=params["epochs"],
              steps_per_epoch=num_training_images // params["batch_size"],
              validation_data=ds_val,
              callbacks=callbacks)

    print("Saved model to ", emb_ckpt_path)
    # emb_model.save_weights(emb_ckpt_path,
    #                        save_format="tf",
    #                        overwrite=True)

    del model, emb_model
    gc.collect()
    # return model, emb_model, optimizer, loss, metrics


def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1


def average_expansion(embeddings, top_k=3):
    norm_emb = tf.math.l2_normalize(embeddings, axis=1)
    sim_matrix = tf.linalg.matmul(norm_emb, norm_emb, transpose_b=True)
    indices = tf.argsort(sim_matrix, direction="DESCENDING")
    top_k_ref_mean = tf.reduce_mean(tf.gather(embeddings, indices[:, :top_k]), axis=1)
    avg_emb = tf.concat([embeddings, top_k_ref_mean])
    return avg_emb


def get_cosine_annealing(params, total_size):
    steps_per_epoch = total_size / params["batch_size"]
    total_steps = int(params["epochs"] * steps_per_epoch)
    warmup_steps = int(params["warmup_epoch"] * steps_per_epoch)

    return WarmUpCosineDecayScheduler(params["lr"], total_steps=total_steps, verbose=params["verbose"],
                                      steps_per_epoch=steps_per_epoch,
                                      warmup_learning_rate=0.0, warmup_steps=warmup_steps, hold_base_rate_steps=0)


def get_linear_decay(params):
    lr_start = 0.000001
    lr_max = 0.000005 * params["batch_size"]
    lr_min = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep = 0
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr

    lr_callback = LearningRateSchedulerPerBatch(lrfn, verbose=True)
    return lr_callback


# def get_linear_decay(params):
#     LR_START = 0.00001
#     LR_MAX = 0.00005 * params["batch_size"]
#     LR_MIN = 0.00001
#     LR_RAMPUP_EPOCHS = 5
#     LR_SUSTAIN_EPOCHS = 0
#     LR_EXP_DECAY = .8
#
#     def lrfn(epoch):
#         if epoch < LR_RAMPUP_EPOCHS:
#             lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
#         elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
#             lr = LR_MAX
#         else:
#             lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
#         return lr
#
#     lr_callback = LearningRateSchedulerPerBatch(lrfn, verbose=True)
#
#     return lr_callback


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ax = plt.subplot()
    t = [get_linear_decay({"batch_size": 128}).schedule(e) for e in range(80)]
    ax.plot(t)
    plt.show()
