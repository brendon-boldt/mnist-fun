import tensorflow as tf
import numpy
np = numpy
from tensorflow.python.ops import nn_ops
import ops

sess = tf.InteractiveSession()

def partial_add(arg_larger, arg_smaller, origin):
  larger = arg_larger
  smaller = arg_smaller
  for i in range(len(smaller)):
    for j in range(len(smaller[0])):
      larger[origin[0]+i,origin[1]+j] += smaller[i,j]
  return larger

def dconv(arg_map, arg_filters, padding=2):
  amap = arg_map#.eval()
  filters = arg_filters#.eval()
  y_size = arg_map.shape[0]#.get_shape().dims[0].value
  x_size = arg_map.shape[1]#.get_shape().dims[1].value
  channels = arg_filters.shape[2]#.get_shape().dims[2].value
  image = numpy.zeros([y_size + padding*2, x_size + padding*2, channels])
  for i in range(y_size):
    for j in range(x_size):
      weighted = filters*amap[i][j]
      '''
        print(weighted.shape)
        exit()
      '''
      delta = numpy.add.reduce(weighted, 3)
      image = partial_add(image, delta, [i, j])
  if padding > 0:
    return image[padding:-padding,padding:-padding]
  else:
    return image

# value [-1, 14, 14, 64]
# fil   [5, 5, 32, 64]
p = np.array([1.0, 2.0, 3.0])
arr = np.array([[p,p],[p,p]])
a_value = np.array([arr, arr, arr, arr])
l = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
a_fil = np.array([[l]])
value = tf.constant(a_value, dtype=tf.float32)
fil   = tf.constant(a_fil, dtype=tf.float32)

#value = np.array([[[[1]],[[2]]],[[[3]],[[4]]]])
#fil = np.array([[[[2]]]])
#result = dconv(value, fil, padding=0)

result = dconv(a_value, a_fil, padding=0)  # nn_ops.deconv2d(value, fil, [-1,2,2,2], [1,1,1,1])
print(result)

#result = ops.partial_add(tf.constant([[1,1,1],[1,1,1],[1,1,1]],dtype=tf.float32), tf.constant([[1,1],[1,1]], dtype=tf.float32), [0,0])
result = ops.conv2d_transpose(value, fil, padding=0)

print(result)
print(result.eval())
