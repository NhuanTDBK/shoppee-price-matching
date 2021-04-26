import numpy as np
# from keras.callbacks import Callback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, mom=0.9, stop_multiplier=None,
                 reload_weights=True, batches_lr_update=5):

        super(LRFinder, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.mom = mom
        self.reload_weights = reload_weights
        self.batches_lr_update = batches_lr_update
        if stop_multiplier is None:
            self.stop_multiplier = -20 * self.mom / 3 + 10  # 4 if mom=0.9
            # 10 if mom=0
        else:
            self.stop_multiplier = stop_multiplier

    def on_train_begin(self, logs={}):
        p = self.params
        try:
            n_iterations = p['epochs'] * p['samples'] // p['batch_size']
        except:
            n_iterations = p['steps'] * p['epochs']

        self.learning_rates = np.geomspace(self.min_lr, self.max_lr, \
                                           num=n_iterations // self.batches_lr_update + 1)
        self.losses = []
        self.iteration = 0
        self.best_loss = 0
        if self.reload_weights:
            self.model.save_weights('tmp.hdf5')

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')

        if self.iteration != 0:  # Make loss smoother using momentum
            loss = self.losses[-1] * self.mom + loss * (1 - self.mom)

        if self.iteration == 0 or loss < self.best_loss:
            self.best_loss = loss

        if self.iteration % self.batches_lr_update == 0:  # Evaluate each lr over 5 epochs

            if self.reload_weights:
                self.model.load_weights('tmp.hdf5')

            K.set_value(self.model.optimizer.lr, self.learning_rates[self.iteration // self.batches_lr_update])
            self.losses.append(loss)

        if loss > self.best_loss * self.stop_multiplier:  # Stop criteria
            self.model.stop_training = True

        self.iteration += 1

    def on_train_end(self, logs=None):
        if self.reload_weights:
            self.model.load_weights('tmp.hdf5')

        plt.figure(figsize=(12, 6))
        plt.plot(self.learning_rates[:len(self.losses)], self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.show()

    def on_epoch_begin(self, epoch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        print('\nEpoch %05d: LearningRateScheduler reducing learning '
              'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


class EarlyStoppingByLossVal(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(EarlyStoppingByLossVal, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        # if current is None:
        #     warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.

    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.

    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.

    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 steps_per_epoch,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,

                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.

    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.

    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []
        print("""
            Warmup step: %s,
            Warmup learning rate: %s
        """%(self.warmup_steps, self.warmup_learning_rate))
        self.steps_per_epoch = steps_per_epoch

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)

    # def on_epoch_begin(self, epoch, logs=None):
    #     print('\n Begin Epoch %05d: setting learning rate to %s.' % (self.global_step + 1,K.get_value(self.model.optimizer.lr)))

    def on_epoch_end(self, epoch, logs=None):
        print('\n After Epoch %05d: setting learning rate to %s.' % (
        self.global_step + 1, K.get_value(self.model.optimizer.lr)))


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_manager: tf.train.CheckpointManager):
        super(CheckpointCallback, self).__init__()
        self.ckpt_manager = ckpt_manager

    def on_epoch_end(self, epoch, logs=None):
        self.ckpt_manager.save(epoch)


if __name__ == '__main__':
    params = {
        "batch_size": 32,
        "epochs": 20,
        "lr": 1e-3,
        "warmup_epoch": 2,
    }
    total_size = 27400
    steps_per_epoch = total_size / params["batch_size"]
    total_steps = int(params["epochs"] * steps_per_epoch)
    warmup_steps = int(params["warmup_epoch"] * steps_per_epoch)

    global_step = 0
    lrs = []
    for _ in range(total_steps):
        lrs.append(cosine_decay_with_warmup(global_step, params["lr"], total_steps, warmup_learning_rate=0.0,
                                            warmup_steps=warmup_steps))
        global_step += 1

    import matplotlib.pyplot as plt

    ax = plt.subplot()
    ax.plot(np.arange(global_step), lrs)
    plt.show()
