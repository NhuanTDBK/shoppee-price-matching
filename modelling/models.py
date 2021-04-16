import tensorflow as tf
from tensorflow.keras import layers, regularizers

from modelling.metrics import ArcFace, AdaCos, CircleLoss, CircleLossCL

from modelling.pooling import *
from features.pool import BertLastHiddenState

metric_layer_dict = {
    "arcface": ArcFace,
    "adacos": AdaCos,
    "circle": CircleLoss,
    "circle_cl": CircleLossCL,
    "linear": layers.Dot

}
pooling_dict = {
    "mac": MAC,
    "gem": GeM,
    "spoc": SPoC,

    "global_avg_1d": layers.GlobalAveragePooling1D,
    "global_max_1d": layers.GlobalMaxPool1D,
}


def _regularizer(weights_decay=5e-4):
    return regularizers.l2(weights_decay)


class TextProductMatch(layers.Layer):
    def get_config(self):
        return {
            "n_classes": self.n_classes
        }

    def __init__(self, n_classes,
                 pooling,
                 use_fc=True,
                 dropout=0.0,
                 metric="arcface",
                 s=24.0,
                 margin=0.3,
                 fc_dim=512,
                 ls_eps=0.0,
                 theta_zero=0.85,
                 is_softmax=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.pooling_layer = pooling_dict[pooling]
        self.use_fc = use_fc
        self.dropout = dropout

        self.s = s
        self.margin = margin
        self.ls_eps = ls_eps
        self.theta_zero = theta_zero
        self.fc_dim = fc_dim
        self.is_softmax = is_softmax

        self.metric_layer = metric_layer_dict[metric](num_classes=self.n_classes,
                                                           margin=self.margin,
                                                           scale=self.s)

        if self.use_fc:
            self.extract_features_layer = tf.keras.Sequential([
                layers.Dropout(self.dropout),
                layers.Dense(self.fc_dim),
                layers.BatchNormalization()
            ])

    def call(self, inputs, training=None, mask=None):
        x, y = inputs

        # x = self.pooling_layer()(x)
        # if self.use_fc:
        #     x = self.extract_features_layer(x)
        x = BertLastHiddenState(multi_sample_dropout=True)(x)

        x = self.metric_layer([x,y])

        x = tf.math.l2_normalize(x, axis=1)

        x = layers.Softmax(dtype="float64")(x)

        return x

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

