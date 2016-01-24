import numpy as np
import tensorflow as tf
import molasses as mo
import sys

sess = tf.InteractiveSession()

filename = sys.argv[1]

with open(filename + '.png', 'rb') as file:
  string = file.read()
  encoded = tf.image.decode_png(tf.constant(string))
  encoded = tf.reshape(encoded, [28,28,1])
  reduced = encoded.eval()/255.0
  mo.array_to_idx(reduced, filename + '.gz')
