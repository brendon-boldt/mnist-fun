from tensorflow.python.ops import nn_ops
import molasses as mo
import numpy as np
import ops
#import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
import tensorflow as tf
sess = tf.InteractiveSession()

batch_size = 1

def weight_variable(shape):
  #initial = tf.abs(tf.truncated_normal(shape, stddev=0.1))
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def generate_image(num):
  ''' Fix command line arguments here
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
    x  = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10 ])
  '''

'''
with open('three.png', 'rb') as file:
  string = file.read()
  encoded = tf.image.decode_png(tf.constant(string))
  encoded = tf.reshape(encoded, [28,28,1])
  print(encoded)
  mo.tensor_to_idx(encoded, '3.gz')
'''

x  = tf.placeholder("float", shape=[None, 784])
'''
  y_ = tf.placeholder("float", shape=[None, 10 ])
  W_fc2 = weight_variable([10, 1024])
  b_fc2 = bias_variable([10]) # I might want to uses biases differently
'''
y_ = tf.placeholder("float", shape=[None, 1])
W_fc2 = weight_variable([1, 1024])
b_fc2 = bias_variable([1]) # I might want to uses biases differently
#h_fc1 = tf.nn.relu(tf.matmul(y_ - b_fc2, W_fc2))
h_fc1 = tf.nn.relu(tf.matmul(y_ + b_fc2, W_fc2))
W_fc1 = weight_variable([1024, 7 * 7 * 64]) # I'm just transposing things :3
b_fc1 = bias_variable([1024])
#h_pool2_flat = tf.matmul(h_fc1 - b_fc1, W_fc1)
h_pool2_flat = tf.matmul(h_fc1 + b_fc1, W_fc1)
h_pool2 = tf.reshape(h_pool2_flat, [-1, 7, 7, 64]) # I think that's right...
h_conv2 = tf.image.resize_images(h_pool2,14,14, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
#h_pool1 = nn_ops.conv2d_transpose(h_conv2 - b_conv2,W_conv2,
h_pool1 = nn_ops.conv2d_transpose(h_conv2 + b_conv2,W_conv2,
    [batch_size,14,14,32],[1,1,1,1])
h_conv1 = tf.image.resize_images(h_pool1 ,28,28,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
#x_image = nn_ops.conv2d_transpose(h_conv1 - b_conv1, W_conv1, [batch_size,28,28,1], [1,1,1,1])
x_image = nn_ops.conv2d_transpose(h_conv1 + b_conv1, W_conv1, [batch_size,28,28,1], [1,1,1,1])
# Is relu good here?
x_image = tf.nn.relu(x_image)
#x_image = x_image + tf.reduce_min(x_image)

########

# Double check that this is acheiving what I want
# It might be better to go with something with something smoother
# Maybe (x - x_image)^2
error = tf.reduce_sum(tf.pow(tf.reshape(x,[-1]) - tf.reshape(x_image, [-1]), 2))
#cross_entropy = -tf.reduce_sum(x * tf.log(x_image))
train_step = tf.train.AdamOptimizer(1e-4).minimize(error)
sess.run(tf.initialize_all_variables())

three_im = np.reshape(np.array([mo.idx_to_array('3.gz')]), [1,-1])/255.0
#three_oh = np.array([0,0,0,1,0,0,0,0,0,0])
three_oh = np.array([[1]])
#fd = {x:three_im}

for i in range(500):
  if i % 20 == 0:
    print("step %d"%(i,))
    print(tf.reduce_max(x_image).eval(feed_dict={y_:three_oh}))
    train_accuracy = error.eval(feed_dict={x:three_im , y_:three_oh})
    print("CE: %s"%(train_accuracy,))
  train_step.run(feed_dict={x:three_im , y_:three_oh})

dir = "result_images/"
with open(dir + "number.png", "wb") as file:
  x_array = (255.0*np.clip(x_image.eval(session=sess, feed_dict={y_:three_oh}), 0.0, 1.0)).astype(np.uint8)
  x_image = tf.constant(x_array, dtype=np.uint8)
  file.write(tf.image.encode_png(tf.squeeze(x_image, [0])).eval(session=sess))

