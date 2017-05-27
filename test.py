
import os.path
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import *
from inputs import *

#import tf.contrib.rnn.BasicConvLSTMCell as BasicConvLSTMCell

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('test_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_string('test_shape', '20000x10000',
                            """ shape of the test image """)

shape = FLAGS.test_shape.split('x')
shape = map(int, shape)

def test():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [1] + shape + [1])

    x_compressed = standard_res_encoder(x)
    x_prime = standard_res_decoding(x_compressed, 1)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = tf.all_variables()
    saver = tf.train.Saver(variables_to_restore)

    # Start running operations on the Graph.
    sess = tf.Session()

    # init from checkpoint
    print("init network from " + FLAGS.test_dir)
    ckpt = tf.train.get_checkpoint_state(FLAGS.test_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    while True:
      dat = make_batch(1, shape)
      print("made batch")
      x_g = sess.run(x_prime,feed_dict={x:dat})
      print("ran one")
      #plt.imshow(np.concatenate([x_g[0,:,:,0], dat[0,:,:,0]], 0))
      #plt.show()


def main(argv=None):  # pylint: disable=unused-argument
  test()

if __name__ == '__main__':
  tf.app.run()


