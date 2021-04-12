import time

import tensorflow as tf

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def measure(x, steps):
  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
  # tf.matmul can return before completing the matrix multiplication
  # (e.g., can return after enqueing the operation on a CUDA stream).
  # The x.numpy() call below will ensure that all enqueued operations
  # have completed (and will also copy the result to host memory,
  # so we're including a little more than just the matmul operation
  # time).
  _ = x.numpy()
  end = time.time()
  return end - start

shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

print("GPU: {} secs".format(measure(tf.random.normal(shape), steps)))
# Run on CPU:
# with tf.device("/cpu:0"):
#   print("CPU: {} secs".format(measure(tf.random.normal(shape), steps)))

# # Run on GPU, if available:
# if tf.config.list_physical_devices("GPU"):
#   with tf.device("/gpu:0"):
#     print("GPU: {} secs".format(measure(tf.random.normal(shape), steps)))
# else:
#   print("GPU: not found")
