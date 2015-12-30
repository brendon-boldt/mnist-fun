import tensorflow as tf
import numpy

sess = tf.InteractiveSession()

def t_map(func, tensor):
  flat = tf.reshape(tensor, [-1]).eval()
  flat_mapped = list(map(func, flat))
  return tf.reshape(tf.constant(flat_mapped), tensor.get_shape())

def t_entropy(tensor):
  log_mapped = t_map(lambda x: x*numpy.log2(x), a)
  return tf.reduce_sum(log_mapped)


a = tf.constant([[1,2],[3,4]])
print(a.eval())
print(t_entropy(a).eval())

