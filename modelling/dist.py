import tensorflow as tf 

@tf.function
def pairwise_dist(x1, x2, eps=1e-6):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x1, x2)+eps)))


if __name__ == "__main__":
    a = tf.constant([[1,2],[4,5]],dtype=tf.float32)
    b = tf.constant([[1,2],[4,5]],dtype=tf.float32)

    print(pairwise_dist(a,b))
    # print(tf.linalg.norm(a-b))