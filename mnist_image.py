import molasses as mo
from tensorflow.python.ops import nn_ops
import numpy as np
import ops
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
import tensorflow as tf
sess = tf.InteractiveSession()

batch_size = 10

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.10)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

'''
  x  = tf.placeholder("float", shape=[None, 784])
  y_ = tf.placeholder("float", shape=[None, 10 ])
  x_image = tf.reshape(x, [-1,28,28,1])

  W_conv1 = weight_variable([5,5,1,32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5,5,32,64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder("float")
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
'''
x  = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10 ])

W_fc2 = weight_variable([10, 1024])
b_fc2 = bias_variable([10]) # I might want to uses biases differently
# Should I be using relu here?
h_fc1 = tf.nn.relu(tf.matmul(y_ - b_fc2, W_fc2))

W_fc1 = weight_variable([1024, 7 * 7 * 64]) # I'm just transposing things :3
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.matmul(h_fc1 - b_fc1, W_fc1)
h_pool2 = tf.reshape(h_pool2_flat, [-1, 7, 7, 64]) # I think that's right...
h_conv2 = tf.image.resize_images(h_pool2,14,14, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
# I'm not sure if the eval's will monkey wrench things

# My conv transpose is rather limited
h_pool1 = ops.conv2d_transpose_batch(h_conv2 - b_conv2, W_conv2, batch_size, padding=2)
#h_pool1 = nn_ops.deconv2d(h_conv2 - b_conv2, W_conv2, [-1,14,14,32], [1,1,1,1])
h_conv1 = tf.image.resize_images(h_pool1 ,28,28, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# BOOOM!
x_image = nn_ops.deconv2d(h_conv1 - b_conv1, W_conv1, [-1,28,28,1], [1,1,1,1])

#x_image = tf.constant(dconv(h_conv1.eval() - b_conv1, W_conv1))
# Do I want dropout at some point here?

########

###
### Change the line below so things are optimized correctly
###
cross_entropy = -tf.reduce_sum(x * tf.log(x_image))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())


'''
batch = mnist.train.next_batch(batch_size)
h_pool1.eval(feed_dict={x: batch[0], y_: batch[1]})
pass
exit()
'''


for i in range(100):
  batch = mnist.train.next_batch(batch_size)
  if i % 1 == 0:
    print("step %d"%(i,))
    '''
    train_accuracy = accuracy.eval(feed_dict={
      x:batch[0], y_:batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    '''
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  print('del')

dir = "result_images/"
with open(dir + "number.png", "wb") as file:
  x_image = tf.constant(ops.normalize(x_image.eval(session=sess, feed_dict={
    y_:[np.array([0,0,0,1,0,0,0,0,0,0])]}), 0, 255))
  #t_n = t_map(lambda x: np.uint8(x), t_n)
  file.write(tf.image.encode_png(x_image).eval(session=sess))

