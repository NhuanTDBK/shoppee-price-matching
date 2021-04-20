from tensorflow.keras import layers, regularizers

from modelling.metrics import ArcFace, AdaCos, CircleLoss, CircleLossCL
from modelling.pooling import *

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
                 dropout=0.0,
                 metric="arcface",
                 s=30.0,
                 margin=0.5,
                 ls_eps=0.0,
                 theta_zero=0.85,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.dropout = dropout

        self.s = s
        self.margin = margin
        self.ls_eps = ls_eps
        self.theta_zero = theta_zero

        self.metric_layer = metric_layer_dict[metric](num_classes=self.n_classes,
                                                      margin=self.margin,
                                                      scale=self.s)

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        x = self.metric_layer([x, y])
        x = layers.Softmax(dtype="float32")(x)
        return x

    def compute_output_shape(self, input_shape):
        return None, self.n_classes
