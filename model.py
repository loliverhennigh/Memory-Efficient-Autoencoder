
import tensorflow as tf
import numpy as np
from nn import *

def standard_res_encoder(inputs, filter_size=2, nr_downsamples=4, nr_residual=2, nonlinearity="concat_elu", gated=True):
  x_i = inputs
  nonlinearity = set_nonlinearity(nonlinearity)

  for i in xrange(nr_downsamples):
    x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=2, gated=gated, name="resnet_down_sampled_" + str(i) + "_nr_residual_0", begin_nonlinearity=False) 
    for j in xrange(nr_residual - 1):
      x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=1, gated=gated, name="resnet_down_sampled_" + str(i) + "_nr_residual_" + str(j+1))
    filter_size = filter_size*2
  print(x_i.get_shape())
  return x_i
standard_res_encoder_template = tf.make_template('standard_res_encoder_state_template', standard_res_encoder)

def conv_lstm_encoder(inputs, nr_slices=4, filter_size=2, nr_downsamples=3, nr_residual=2, nonlinearity="concat_elu", gated=True):
  inputs = tf.split(inputs, nr_slices, 1)
  hidden_1 = None
  hidden_2 = None
  for i in xrange(nr_slices):
    x_conv_compression = standard_res_encoder_template(inputs[i][:,0], filter_size=filter_size, nr_downsamples=nr_downsamples, nr_residual=nr_residual, nonlinearity=nonlinearity, gated=gated)
    x_conv_lstm_residual, hidden_1, hidden_2 = res_block_lstm_template(x_conv_compression, hidden_1, hidden_2)
  print(hidden_1.get_shape())
  return hidden_1, hidden_2

def standard_res_decoder(inputs, final_size=1, filter_size=2, nr_downsamples=4, nr_residual=2, nonlinearity="concat_elu", gated=True):
  y_i = inputs
  nonlinearity = set_nonlinearity(nonlinearity)
  filter_size = filter_size*pow(2,nr_downsamples-2)
  for i in xrange(nr_downsamples-1):
    y_i = transpose_conv_layer(y_i, 4, 2, filter_size, "up_conv_" + str(i))
    for j in xrange(nr_residual):
      y_i = res_block(y_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=1, gated=gated, name="resnet_up_sampled_" + str(i) + "_nr_residual_" + str(j+1))
    filter_size = filter_size/2
  y_i = transpose_conv_layer(y_i, 4, 2, final_size, "up_conv_" + str(nr_downsamples))
  return tf.nn.tanh(y_i)
standard_res_decoder_template = tf.make_template('standard_res_decoder_state_template', standard_res_decoder)

def conv_lstm_decoder(hidden_1, hidden_2, nr_slices=4, filter_size=2, nr_downsamples=3, nr_residual=2, nonlinearity="concat_elu", gated=True):
  x_out = []
  x_conv_lstm_residual = tf.zeros_like(tf.split(hidden_1, 2, 3)[0])
  for i in xrange(nr_slices):
    x_conv_lstm_residual, hidden_1, hidden_2 = res_block_lstm_template(x_conv_lstm_residual, hidden_1, hidden_2)
    x_out.append(standard_res_decoder_template(x_conv_lstm_residual, filter_size=filter_size, nr_downsamples=nr_downsamples, nr_residual=nr_residual, nonlinearity=nonlinearity, gated=gated))
  x_out = tf.stack(x_out, axis=1)
  return x_out



