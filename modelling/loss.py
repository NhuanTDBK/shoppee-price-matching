import tensorflow as tf

@tf.function
def contrastive_loss(y_true, y_pred, eps=1e-07, margin=0.7):
    # y_pred = tf.convert_to_tensor(y_pred)
    # y_true = tf.cast(y_true, y_pred.dtype)
    
    # D = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred + eps),))

    # return 0.5 * y_true * tf.math.square(D) + 0.5 * (1-y_true) * tf.math.square(
    #         tf.math.maximum(0, margin -  y_pred)
    # )
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    loss = 0.5 * y_true * tf.math.square(y_pred) + 0.5 * (1.0 - y_true) * tf.math.square(tf.math.maximum(margin - y_pred, 0.0))
    return tf.reduce_sum(loss)

class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin =0.7, eps=1e-07, reduction=tf.keras.losses.Reduction.SUM):
        self.margin = margin
        self.eps = eps
    
    def call(self, y_true, y_pred):
        return tf.reduce_sum(contrastive(y_true,y_pred,self.eps, self.margin))

if __name__ == "__main__":
    
    a = tf.constant([[1, 2],[3, 4],[5, 6]], dtype=tf.float16)
    b = tf.constant([[5, 9],[3, 6],[1, 8]], dtype=tf.float16)
    y_pred = tf.linalg.norm(a - b, axis=1)
    y_true = tf.constant([0,1,0],dtype=tf.int8)

    t = ContrastiveLoss()
    print(t(y_true, y_pred))


