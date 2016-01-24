from tensorflow.python.ops import nn_ops
import molasses as mo
import numpy as np
import ops
import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.05)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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

# For now, I think the batch size should equal the class size
#batch_size = 1
class_size = 10
fc_size = 1024
conv1_size = 32
conv2_size = 64


# Figure where I should use batch_size and where I should use class_size
x  = tf.placeholder("float", shape=[class_size, 784])
'''
  y_ = tf.placeholder("float", shape=[None, 10 ])
  W_fc2 = weight_variable([10, 1024])
  b_fc2 = bias_variable([10]) # I might want to uses biases differently
'''
y_ = tf.placeholder("float", shape=[class_size, class_size])
W_fc2 = weight_variable([class_size, fc_size])
b_fc2 = bias_variable([class_size])
h_fc1 = tf.nn.relu(tf.matmul(y_ + b_fc2, W_fc2))
W_fc1 = weight_variable([fc_size, 7 * 7 * conv2_size])
b_fc1 = bias_variable([fc_size])
h_pool2_flat = tf.matmul(h_fc1 + b_fc1, W_fc1)
h_pool2 = tf.reshape(h_pool2_flat, [-1, 7, 7, conv2_size]) # I think that's right...
h_conv2 = tf.image.resize_images(h_pool2,14,14, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
W_conv2 = weight_variable([5,5,conv1_size,conv2_size])
b_conv2 = bias_variable([conv2_size])
'''
  print(h_conv2.get_shape())
  print((h_conv2 + b_conv2).get_shape())
  print([class_size,14,14,conv1_size])
  exit()
'''
h_pool1 = nn_ops.conv2d_transpose(h_conv2 + b_conv2,W_conv2,
    [class_size,14,14,conv1_size],[1,1,1,1])
h_conv1 = tf.image.resize_images(h_pool1 ,28,28,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
W_conv1 = weight_variable([5,5,1,conv1_size])
b_conv1 = bias_variable([conv1_size])
x_image = nn_ops.conv2d_transpose(h_conv1 + b_conv1, W_conv1, [class_size,28,28,1], [1,1,1,1])
x_image = tf.nn.relu(x_image)

########

# Do these have to be changed based on class size?
error = tf.reduce_sum(tf.pow(tf.reshape(x,[-1]) - tf.reshape(x_image, [-1]), 2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(error)
sess.run(tf.initialize_all_variables())

# Put the file names of the images in order
image_names = ['three.gz'] * class_size
images = np.array(list(map(lambda x: mo.idx_to_array(x), image_names)))
images = np.reshape(images, [class_size, 28*28])
#images = np.reshape(np.array([mo.idx_to_array('three.gz')]), [1,-1])/255.0
#oh_encodings = np.array([[1]])
oh_encodings = np.identity(class_size)
#fd = {x:three_im}

for i in range(30):
  if i % 20 == 0:
    print("step %d"%(i,))
    #print(tf.reduce_max(x_image).eval(feed_dict={y_:[oh_encodings[0]]}))
    train_accuracy = error.eval(feed_dict={x:images , y_:oh_encodings})
    print("CE: %s"%(train_accuracy,))
  train_step.run(feed_dict={x:images , y_:oh_encodings})

exit()
dir = "result_images/"
with open(dir + "number.png", "wb") as file:
  x_array = (255.0*np.clip(x_image.eval(session=sess, feed_dict={y_:three_oh}), 0.0, 1.0)).astype(np.uint8)
  x_image = tf.constant(x_array, dtype=np.uint8)
  file.write(tf.image.encode_png(tf.squeeze(x_image, [0])).eval(session=sess))


exit()
#W_conv1_t = tf.transpose(W_conv1, perm=[3,0,1,2])
#W_conv2_t = tf.reduce_sum(tf.transpose(W_conv2, perm=[3,0,1,2]), [3], keep_dims=True)

index = 0
dir = "filters/"
#for t in W_conv1_t.eval():
for t in W_conv2_t.eval():
  with open(dir+'filter'+str(index)+'.png', "wb") as file:
    t = tf.constant(t)
    t_n = tf.image.resize_images(t,50,50, method=tf.image.ResizeMethod.BICUBIC)
    t_n = tf.constant(ops.normalize(t_n.eval(), 0, 255))
    #t_n = t_map(lambda x: numpy.uint8(x), t_n)
    file.write(tf.image.encode_png(t_n).eval())
  index += 1
