
import os.path
import time

import numpy as np
import tensorflow as tf

from model import *
from inputs import *
from utils import *

#import tf.contrib.rnn.BasicConvLSTMCell as BasicConvLSTMCell

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('max_step', 20000,
                            """max num of steps""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_string('shape', '256x256',
                            """ shape of the test image """)
tf.app.flags.DEFINE_float('lr', 0.0003,
                            """ learning rate """)

shape = FLAGS.shape.split('x')
shape = map(int, shape)

def train():
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, 4, shape[0]/2, shape[1]/2, 1])

    h_1, h_2 = conv_lstm_encoder(x)
    x_prime = conv_lstm_decoder(h_1, h_2)

    # calc total loss
    for i in xrange(4):
      tf.summary.image('true_x_' + str(i), x[:,i])
      tf.summary.image('generated_x_' + str(i), x_prime[:,i])
    loss = tf.nn.l2_loss(x - x_prime)
    tf.summary.scalar('loss', loss)

    # training
    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    
    # List of all Variables
    variables = tf.global_variables()

    # Build a saver
    saver = tf.train.Saver(tf.global_variables())   

    # Summary op
    summary_op = tf.summary.merge_all()
 
    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)

    for step in xrange(FLAGS.max_step):
      dat = make_batch(FLAGS.batch_size, shape)
      dat = image_to_grid(dat, shape, 2)
      t = time.time()
      _, loss_r = sess.run([train_op, loss],feed_dict={x:dat})
      elapsed = time.time() - t

      if step%100 == 0 and step != 0:
        summary_str = sess.run(summary_op, feed_dict={x:dat})
        summary_writer.add_summary(summary_str, step) 
        print("time per batch is " + str(elapsed))
        print(step)
        print(loss_r)
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()


