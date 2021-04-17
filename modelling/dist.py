import tensorflow as tf


@tf.function
def pairwise_dist(x1, x2, eps=1e-6):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x1, x2) + eps)))


class ManDist(tf.keras.layers.Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = tf.math.exp(-tf.reduce_sum(tf.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return tf.shape(self.result)


if __name__ == "__main__":
    a = tf.constant([[1, 2], [4, 5]], dtype=tf.float32)
    b = tf.constant([[1, 2], [4, 5]], dtype=tf.float32)

    print(pairwise_dist(a, b))
    # print(tf.linalg.norm(a-b))
