import tensorflow as tf
from modelling.models import  TextProductMatch

class MultiheadAttentionTest(tf.test.TestCase):

    def testScaledDotProductAttentionOutputCorrectness(self):
        batch_size = 1
        seq_len = 2
        depth = 1
        X = tf.ones([batch_size, seq_len, depth])
        y = tf.ones(1)

        input = tf.keras.layers.Input(shape=(3))
        model = TextProductMatch(12,"gem")(input)
        model.build()
        print(model.summary())

        # model.fit(X,y)



if __name__ == '__main__':
    tf.test.main()