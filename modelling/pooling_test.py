import tensorflow as tf

from modelling import pooling


class PoolingsTest(tf.test.TestCase):

    def testMac(self):
        x = tf.constant([[[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]]])
        # Run tested function.
        result = pooling.mac(x)
        # Define expected result.
        exp_output = [[6., 7.]]
        # Compare actual and expected.
        self.assertAllClose(exp_output, result)

    def testSpoc(self):
        x = tf.constant([[[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]]])
        # Run tested function.
        result = pooling.spoc(x)
        # Define expected result.
        exp_output = [[3., 4.]]
        # Compare actual and expected.
        self.assertAllClose(exp_output, result)

    def testGem(self):
        x = tf.constant([[[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]]])
        # Run tested function.
        result = pooling.gem(x, power=3., eps=1e-6)
        # Define expected result.
        exp_output = [[4.1601677, 4.9866314]]
        # Compare actual and expected.
        self.assertAllClose(exp_output, result)


if __name__ == '__main__':
    tf.test.main()
