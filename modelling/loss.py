import tensorflow as tf


@tf.function
def contrastive_loss(y_true, y_pred, margin=0.7, agg=1):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)

    loss = 0.5 * y_true * tf.math.square(y_pred) + 0.5 * (1.0 - y_true) * tf.math.square(
        tf.math.maximum(margin - y_pred, 0.0))
    # return tf.cond(tf.equal(agg,1),tf.reduce_sum(loss),tf.reduce_mean(loss))
    return tf.reduce_sum(loss) if agg == 1 else tf.reduce_mean(loss)


if __name__ == "__main__":
    a = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
    b = tf.constant([[5, 9], [3, 6], [1, 8]], dtype=tf.float16)
    y_pred = tf.linalg.norm(a - b, axis=1)
    y_true = tf.constant([0, 1, 0], dtype=tf.int8)
    print(contrastive_loss(a, b, 1))
    print(contrastive_loss(a, b, 2))

    # t = ContrastiveLoss()
    # print(t(y_true, y_pred))
