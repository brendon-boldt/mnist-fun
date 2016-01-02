import molasses as mo
import numpy
import time
import ops

import tensorflow as tf
sess = tf.InteractiveSession()

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

# amap      : [14,14,64]
# filters   : [5,5,32,64]
# Filter format [x,y,channels,volume]
def dconv(arg_map, arg_filters, padding=2):
  amap = arg_map.eval()
  filters = arg_filters.eval()
  y_size = arg_map.get_shape().dims[0].value
  x_size = arg_map.get_shape().dims[1].value
  channels = arg_filters.get_shape().dims[2].value
  #image = tf.zeros([y_size + padding*2, x_size + padding*2, channels])
  image = numpy.zeros([y_size + padding*2, x_size + padding*2, channels])
  for i in range(y_size):
    for j in range(x_size):
      weighted = filters*amap[i][j]
      delta = numpy.add.reduce(weighted, 3)
      image = partial_add(image, delta, [i, j])
  return tf.constant(image[padding:-padding,padding:-padding])

dir = "weights/"
#y_ = tf.placeholder("float", shape=[None, 10 ])
# 1-hot encoding for the number 0 (or maybe 9)
# May have to change it to hardmin it (un-softmax)
W_fc2 = mo.idx_to_tensor(dir + "W_fc2")
b_fc2 = mo.idx_to_tensor(dir + "b_fc2")
W_fc1 = mo.idx_to_tensor(dir + "W_fc1")
b_fc1 = mo.idx_to_tensor(dir + "b_fc1")
b_conv2 = mo.idx_to_tensor(dir + "b_conv2")
W_conv2 = mo.idx_to_tensor(dir + "W_conv2")
b_conv1 = mo.idx_to_tensor(dir + "b_conv1")
W_conv1 = mo.idx_to_tensor(dir + "W_conv1")

'''
  Math! Find x!

  h_conv1 = conv(x, W_conv1) + b_conv1
  h_pool1 = pool(h_conv1)
  h_conv2 = conv(h_pool1, W_conv2) + b_conv2
  h_pool2 = pool(h_conv2)
  h_fc1 = relu(h_pool2_flat * W_fc1 + b_fc1))
  y_conv = h_fc1 * W_fc2 + b_fc2

  h_fc1 = (y_conv-b_fc2) * ^W_fc2
  h_pool2 = (h_fc1-b_fc1) * ^W_fc1
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
  And we're probably going to need to introduce hyperparameters
'''
# Not sure if I need relu here, but it seems like a good idea
'''
H_y_ = 5
y_ = H_y_ * tf.constant([[0,0,0,1,0,0,0,0,0,0]], dtype=numpy.float32)
h_fc1 = tf.nn.relu(tf.matmul(y_ - b_fc2, tf.transpose(W_fc2)))

'''

h_fc1 = mo.idx_to_tensor(dir+'h_fc1')
h_pool2_flat = tf.matmul((h_fc1 - b_fc1), numpy.linalg.pinv(W_fc1.eval()))
h_pool2 = tf.reshape(h_pool2_flat, [-1, 64, 1]) # I think that's right...
h_conv2 = tf.image.resize_images(h_pool2,14,14, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
h_pool1 = dconv(h_conv2 - b_conv2, W_conv2)
h_conv1 = tf.image.resize_images(h_pool1 ,28,28, method=tf.image.ResizeMethod.BICUBIC)
#h_conv1 = tf.squeeze(mo.idx_to_tensor(dir+'h_conv1'))
x = dconv(h_conv1 - b_conv1, W_conv1)
#print(tf.squeeze(x).eval())
#x = tf.squeeze(mo.idx_to_tensor(dir+'x_image'), [0])

'''
  print("x_image\t", tf.reduce_max(x).eval())
  print("h_conv1\t", tf.reduce_max(h_conv1).eval())
  print("h_pool1\t", tf.reduce_max(h_pool1).eval())
  print("h_conv2\t", tf.reduce_max(h_conv2).eval())
  print("h_pool2\t", tf.reduce_max(h_pool2).eval())
  print("h_fc1\t", tf.reduce_max(h_fc1).eval())
  print("y_conv\t", tf.reduce_max(y_).eval())
'''

with open("number.png", "wb") as file:
  x = tf.constant(ops.normalize(x.eval(), 0, 255))
  #t_n = t_map(lambda x: numpy.uint8(x), t_n)
  file.write(tf.image.encode_png(x).eval())


### Compare each step with a properly formed one form mnist_reverse.py
### Also, I am not even sure if deconv works properly, I just kind of assumed it did
