import sys
import time
import molasses as mo
import numpy
import ops

import tensorflow as tf
global_sess = tf.InteractiveSession()

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

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
      delta = numpy.add.reduce(weighted, 3)
      image = partial_add(image, delta, [i, j])
  return image[padding:-padding,padding:-padding]

if len(sys.argv) > 1:
  dir = sys.argv[1].strip('/') + '/'
else:
  dir = "weights/"
_W_fc2 = mo.idx_to_array(dir + "W_fc2")
_W_fc2_pinv = numpy.linalg.pinv(_W_fc2, 1e-7).astype('f4')
_b_fc2 = mo.idx_to_array(dir + "b_fc2")
_W_fc1 = mo.idx_to_array(dir + "W_fc1")
_W_fc1_pinv = mo.idx_to_array(dir + "W_fc1_pinv").astype('f4')
_b_fc1 = mo.idx_to_array(dir + "b_fc1")
_b_conv2 = mo.idx_to_array(dir + "b_conv2")
_W_conv2 = mo.idx_to_array(dir + "W_conv2")
_b_conv1 = mo.idx_to_array(dir + "b_conv1")
_W_conv1 = mo.idx_to_array(dir + "W_conv1")

'''
  h_conv1 = conv(x, W_conv1) + b_conv1
  h_pool1 = pool(h_conv1)
  h_conv2 = conv(h_pool1, W_conv2) + b_conv2
  h_pool2 = pool(h_conv2)
  h_fc1 = relu(h_pool2_flat * W_fc1 + b_fc1))
  y_conv = h_fc1 * W_fc2 + b_fc2

  h_fc1 = (y_conv-b_fc2) * W_fc2+
  h_pool2 = (h_fc1-b_fc1) * W_fc1+
  h_conv2 = depool(h_pool2)
  h_pool1 = deconv(h_conv2 - b_conv2, W_conv2)
  h_conv1 = depool(h_pool1)
  x = deconv(h_conv1 - b_conv1, W_conv1)
'''

'''
  The problem with this part is that we cannot know the 'proper'
  activation values that got us the (given) y_; it should be enough, though
  to assume that all activation happened in proporition to the weights of
  the full connected layer.
  And we're probably going to need to introduce hyperparameters.
  It seems as if hyperparameters only go so far as to make things more constrasted
  as opposed to actually refining any of the present data.
'''
def reverse_mnist(number):
  with tf.Graph().as_default() as graph:
    with tf.Session() as sess:
      H_y_ = 50
      y_ = H_y_ * tf.constant(numpy.array([numpy.roll([1.0,0,0,0,0,0,0,0,0,0], number)]), dtype=numpy.float32)
      h_fc1 = 5 * tf.nn.relu(tf.matmul(y_ - _b_fc2, numpy.transpose(_W_fc2).astype('f4')))
      h_pool2_flat =tf.matmul(h_fc1 - _b_fc1, _W_fc1_pinv)

      # It figures that the line labeled with "I think that's right" is the problem
      h_pool2 = tf.reshape(h_pool2_flat, [7, 7, 64]) # I think that's right...
      h_conv2 = tf.image.resize_images(h_pool2,14,14, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      h_pool1 = tf.constant(dconv(h_conv2.eval() - _b_conv2, _W_conv2))
      h_conv1 = tf.image.resize_images(h_pool1 ,28,28, method=tf.image.ResizeMethod.BICUBIC)
      x = tf.constant(dconv(h_conv1.eval() - _b_conv1, _W_conv1))
      with open("number"+str(number)+".png", "wb") as file:
        x = tf.constant(ops.normalize(x.eval(session=sess), 0, 255))
        #t_n = t_map(lambda x: numpy.uint8(x), t_n)
        file.write(tf.image.encode_png(x).eval(session=sess))

for i in range(10):
  reverse_mnist(i)
