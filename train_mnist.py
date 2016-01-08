import sys
import molasses as mo
import numpy

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

import tensorflow as tf
sess = tf.InteractiveSession()

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

def write_weights():
  dir = "weights/"
  mo.tensor_to_idx(W_fc1, dir+"W_fc1")
  mo.tensor_to_idx(b_fc1, dir+"b_fc1")
  mo.tensor_to_idx(W_fc2, dir+"W_fc2")
  mo.tensor_to_idx(b_fc2, dir+"b_fc2")
  mo.tensor_to_idx(W_conv1, dir+"W_conv1")
  mo.tensor_to_idx(b_conv1, dir+"b_conv1")
  mo.tensor_to_idx(W_conv2, dir+"W_conv2")
  mo.tensor_to_idx(b_conv2, dir+"b_conv2")
  '''
  mo.tensor_to_idx(W_fc1.eval(), dir+"W_fc1")
  mo.tensor_to_idx(b_fc1.eval(), dir+"b_fc1")
  mo.tensor_to_idx(W_fc2.eval(), dir+"W_fc2")
  mo.tensor_to_idx(b_fc2.eval(), dir+"b_fc2")
  mo.tensor_to_idx(W_conv1.eval(), dir+"W_conv1")
  mo.tensor_to_idx(b_conv1.eval(), dir+"b_conv1")
  mo.tensor_to_idx(W_conv2.eval(), dir+"W_conv2")
  mo.tensor_to_idx(b_conv2.eval(), dir+"b_conv2")
  '''


if len(sys.argv) > 1 and sys.argv[1] == 'new':
  W_conv1 = weight_variable([5,5,1,32])
  b_conv1 = bias_variable([32])
  W_conv2 = weight_variable([5,5,32,64])
  b_conv2 = bias_variable([64])
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  sess.run(tf.initialize_all_variables()) 
  write_weights()


dir = "weights/"

x  = tf.placeholder("float", shape=[None, 784])
y_= tf.placeholder("float", shape=[None, 10 ])
x_image = tf.reshape(x, [-1,28,28,1])

# W_conv1 = weight_variable([5,5,1,32])
# b_conv1 = bias_variable([32])
W_conv1 = tf.Variable(mo.idx_to_tensor(dir+'W_conv1'))
b_conv1 = tf.Variable(mo.idx_to_tensor(dir+'b_conv1'))
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = tf.Variable(mo.idx_to_tensor(dir+'W_conv2'))
b_conv2 = tf.Variable(mo.idx_to_tensor(dir+'b_conv2'))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = tf.Variable(mo.idx_to_tensor(dir+'W_fc1'))
b_fc1 = tf.Variable(mo.idx_to_tensor(dir+'b_fc1'))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(mo.idx_to_tensor(dir+'W_fc2'))
b_fc2 = tf.Variable(mo.idx_to_tensor(dir+'b_fc2'))
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables()) 

#### Add a way to set the number of loops
#### Also create a way to safely interrrupt the process
## Also, this is going to lead to overfitting, just to let you know
if len(sys.argv) > 1 and sys.argv[1] == 'placeholders':
  dir = 'weights/'
  batch = mnist.train.next_batch(50)
  mo.tensor_to_idx(x_image.eval(feed_dict={x:[batch[0][0]]}), dir+"x_image")
  mo.tensor_to_idx(h_conv1.eval(feed_dict={x:[batch[0][0]]}), dir+"h_conv1")
  mo.tensor_to_idx(h_pool1.eval(feed_dict={x:[batch[0][0]]}), dir+"h_pool1")
  mo.tensor_to_idx(h_conv2.eval(feed_dict={x:[batch[0][0]]}), dir+"h_conv2")
  mo.tensor_to_idx(h_pool2.eval(feed_dict={x:[batch[0][0]]}), dir+"h_pool2")
  mo.tensor_to_idx(h_fc1.eval(feed_dict={x:[batch[0][0]]}), dir+"h_fc1")
  mo.tensor_to_idx((tf.matmul(h_fc1_drop, W_fc2) + b_fc2).eval(feed_dict={x:[batch[0][0]], keep_prob:1.0}), dir+"y_conv")
  print("x_image\t", tf.reduce_max(x_image).eval(feed_dict={x:[batch[0][0]]}))
  print("h_conv1\t", tf.reduce_max(h_conv1).eval(feed_dict={x:[batch[0][0]]}))
  print("h_pool1\t", tf.reduce_max(h_pool1).eval(feed_dict={x:[batch[0][0]]}))
  print("h_conv2\t", tf.reduce_max(h_conv2).eval(feed_dict={x:[batch[0][0]]}))
  print("h_pool2\t", tf.reduce_max(h_pool2).eval(feed_dict={x:[batch[0][0]]}))
  print("h_fc1\t", tf.reduce_max(h_fc1).eval(feed_dict={x:[batch[0][0]]}))
  print("y_conv\t", tf.reduce_max(tf.matmul(h_fc1_drop, W_fc2) + b_fc2).eval(feed_dict={x:[batch[0][0]], keep_prob:1.0}))
elif len(sys.argv) > 2 and sys.argv[1] == 'number':
  x_temp = [tf.reshape(mo.idx_to_tensor("idx_numbers/"+sys.argv[2]+".gz"), [-1]).eval()]
  y_temp = numpy.array([[1,0,0,0,0,0,0,0,0,0]], dtype=numpy.float32)
  print(y_conv.eval(feed_dict={x:x_temp, y_:y_temp, keep_prob:1.0}))
else:
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      write_weights()
      train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_:batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

