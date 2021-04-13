from enum import Enum

import numpy as np
import tensorflow as tf


class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    CONCATENATION = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()


class BertLastHiddenState(tf.keras.layers.Layer):
    def __init__(self, last_hidden_states=3, mode=PoolingStrategy.REDUCE_MEAN_MAX, fc_dim=512,
                 multi_sample_dropout=False):
        super(BertLastHiddenState, self).__init__()

        self.last_hidden_states = last_hidden_states
        self.mode = mode
        self.fc_dim = fc_dim

        self.fc = tf.keras.models.Sequential(tf.keras.layers.Dense(self.fc_dim))
        self.multi_sample_dropout = multi_sample_dropout

    def call(self, inputs):
        x = inputs

        x1 = tf.concat([x[-i - 1] for i in range(self.last_hidden_states)], axis=-1)
        if self.mode == PoolingStrategy.REDUCE_MEAN_MAX:
            x1_mean = tf.math.reduce_mean(x1, axis=1)
            x1_max = tf.math.reduce_max(x1, axis=1)
            x_pool = tf.concat([x1_mean, x1_max], axis=1)
        elif self.mode == PoolingStrategy.CONCATENATION:
            return x1
        elif self.mode == PoolingStrategy.REDUCE_MAX:
            x_pool = tf.math.reduce_max(x1, axis=1)
        elif self.mode == PoolingStrategy.REDUCE_MEAN:
            x_pool = tf.math.reduce_mean(x1, axis=1)

        if self.multi_sample_dropout:
            dense_fc = []
            for p in np.linspace(0.1, 0.5, 5):
                x1 = tf.keras.layers.Dropout(p)(x_pool)
                x1 = self.fc(x1)
                dense_fc.append(x1)

            out = tf.keras.layers.Average()(dense_fc)
        else:
            out = self.fc(x_pool)

        return out
