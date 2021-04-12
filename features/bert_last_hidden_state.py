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

class BertLastHiddenState(layers.Layer):
    def __init__(self, name, last_hidden_states=3, mode=PoolingStrategy.REDUCE_MEAN, fc_dim = 512):
        self.name = name
        self.last_hidden_states = last_hidden_states
        self.mode = mode
        self.fc_dim = fc_dim

        self.dense = Dense(self.fc_dim)
        self.dropout_rates = np.array([0.3,0.4,0.5,0.6])

    def build(self, input_shape):
        self.dense.build(input_shape)

    def call(self, inputs):
        x, y = inputs

        x1 = tf.concat([x[-i-1] for i in range(self.last_hidden_states)],axis=-1)
        if self.mode == PoolingStrategy.REDUCE_MEAN_MAX:
            x1_mean = tf.math.reduce_mean(x1, axis=1)
            x1_max = tf.math.reduce_max(x1, axis=1)
            x1 = tf.concat([x1_mean, x1_max],axis=1)
        elif self.mode == PoolingStrategy.CONCATENATION:
            return x1
        elif self.mode == PoolingStrategy.REDUCE_MAX:
            x1 = tf.math.reduce_max(x1, axis=1)
        elif self.mode == PoolingStrategy.REDUCE_MEAN:
            x1 = tf.math.reduce_mean(x1, axis=1)

        return x1
        
        

            

