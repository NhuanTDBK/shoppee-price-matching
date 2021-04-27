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
                 multi_sample_dropout=True):
        super(BertLastHiddenState, self).__init__()

        self.last_hidden_states = last_hidden_states
        self.mode = mode
        self.fc_dim = fc_dim

        self.fc = None
        if fc_dim:
            self.fc = tf.keras.models.Sequential(tf.keras.layers.Dense(self.fc_dim, name="bert_fc"))
        self.multi_sample_dropout = multi_sample_dropout

    def call(self, inputs):
        x = inputs

        x_pool = None

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

        if self.multi_sample_dropout and self.fc_dim:
            dense_fc = []
            for p in np.linspace(0.1, 0.5, 5):
                x1 = tf.keras.layers.Dropout(p,name="dropout_bert")(x_pool)
                x1 = self.fc(x1)
                dense_fc.append(x1)

            out = tf.keras.layers.Average(name="avg_bert")(dense_fc)
        elif not self.multi_sample_dropout and self.fc_dim is not None:
            out = self.fc(x_pool)
        else:
            out = x_pool

        return out


class MAC(tf.keras.layers.Layer):
    """Global max pooling (MAC) layer.
     Maximum Activations of Convolutions (MAC) is simply constructed by
     max-pooling over all dimensions per feature map. See
     https://arxiv.org/abs/1511.05879 for a reference.
    """

    def call(self, x, axis=None):
        """Invokes the MAC pooling instance.
        Args:
          x: [B, H, W, D] A float32 Tensor.
          axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
        Returns:
          output: [B, D] A float32 Tensor.
        """
        if axis is None:
            axis = [1, 2]
        return mac(x, axis=axis)


class SPoC(tf.keras.layers.Layer):
    """Average pooling (SPoC) layer.
    Sum-pooled convolutional features (SPoC) is based on the sum pooling of the
    deep features. See https://arxiv.org/pdf/1510.07493.pdf for a reference.
    """

    def call(self, x, axis=None):
        """Invokes the SPoC instance.
        Args:
          x: [B, H, W, D] A float32 Tensor.
          axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
        Returns:
          output: [B, D] A float32 Tensor.
        """
        if axis is None:
            axis = [1, 2]
        return spoc(x, axis)


class GeM(tf.keras.layers.Layer):
    """Generalized mean pooling (GeM) layer.
    Generalized Mean Pooling (GeM) computes the generalized mean of each
    channel in a tensor. See https://arxiv.org/abs/1711.02512 for a reference.
    """

    def __init__(self, power=3.):
        """Initialization of the generalized mean pooling (GeM) layer.
        Args:
          power:  Float power > 0 is an inverse exponent parameter, used during the
            generalized mean pooling computation. Setting this exponent as power > 1
            increases the contrast of the pooled feature map and focuses on the
            salient features of the image. GeM is a generalization of the average
            pooling commonly used in classification networks (power = 1) and of
            spatial max-pooling layer (power = inf).
        """
        super(GeM, self).__init__()
        self.power = power
        self.eps = 1e-6

    def call(self, x, axis=None):
        """Invokes the GeM instance.
        Args:
          x: [B, H, W, D] A float32 Tensor.
          axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
        Returns:
          output: [B, D] A float32 Tensor.
        """
        if axis is None:
            axis = [1, 2]
        return gem(x, power=self.power, eps=self.eps, axis=axis)


def mac(x, axis=None):
    """Performs global max pooling (MAC).
    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
    Returns:
      output: [B, D] A float32 Tensor.
    """
    if axis is None:
        axis = [1, 2]
    return tf.reduce_max(x, axis=axis, keepdims=False)


def spoc(x, axis=None):
    """Performs average pooling (SPoC).
    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
    Returns:
      output: [B, D] A float32 Tensor.
    """
    if axis is None:
        axis = [1, 2]
    return tf.reduce_mean(x, axis=axis, keepdims=False)


def gem(x, axis=None, power=3., eps=1e-6):
    """Performs generalized mean pooling (GeM).
    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
      power: Float, power > 0 is an inverse exponent parameter (GeM power).
      eps: Float, parameter for numerical stability.
    Returns:
      output: [B, D] A float32 Tensor.
    """
    if axis is None:
        axis = [1, 2]
    tmp = tf.pow(tf.maximum(x, eps), power)
    out = tf.pow(tf.reduce_mean(tmp, axis=axis, keepdims=False), 1. / power)
    return out


pooling_dict = {
    "mac": MAC,
    "gem": GeM,
    "spoc": SPoC,

    "global_avg_1d": tf.keras.layers.GlobalAveragePooling1D,
    "global_max_1d": tf.keras.layers.GlobalMaxPool1D,
}


class LocalGlobalExtractor(tf.keras.layers.Layer):
    def __init__(self, pool, fc_dim=512, dropout_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fts = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dropout(dropout_rate,name="dropout_lb"),
                tf.keras.layers.Dense(fc_dim,name="fc_lb"),
                tf.keras.layers.BatchNormalization(name="bn_lb")
            ])
        self.pool_layer = pooling_dict[pool]()

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.pool_layer(x)
        x = self.fts(x)

        return x
