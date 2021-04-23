import matplotlib.pyplot as plt
# from keras.callbacks import Callback
import numpy as np
import tensorflow as tf


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

            self.model.optimizer.lr = self.learning_rates[self.iteration // self.batches_lr_update]

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
