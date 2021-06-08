import tensorflow as tf
from modelling.loss import ContrastiveLoss

if __name__ == "__main__":
    
    a = tf.constant([[0.1, 0.2],[0.3, 0.4],[0.5, 0.6]], dtype=tf.float16)
    b = tf.constant([[0.5, 0.9],[0.3, 0.6],[1, 0.88]], dtype=tf.float16)
    y_pred = tf.linalg.norm(a - b, axis=1)
    y_true = tf.constant([0,1,0],dtype=tf.int8)

    t = ContrastiveLoss()
    print(t.call(y_true, ),.numpy())