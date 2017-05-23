
import tensorflow as tf
import numpy as np
from nn import *

def standard_res_encoder(inputs, filter_size=2, nr_downsamples=5, nr_residual=2, nonlinearity="concat_elu", gated=True):
  x_i = inputs
  nonlinearity = set_nonlinearity(nonlinearity)

  for i in xrange(nr_downsamples):
    print(filter_size)
    x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=2, gated=gated, name="resnet_down_sampled_" + str(i) + "_nr_residual_0", begin_nonlinearity=False) 
    for j in xrange(nr_residual - 1):
      x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=1, gated=gated, name="resnet_down_sampled_" + str(i) + "_nr_residual_" + str(j+1))
    filter_size = filter_size*2
  return x_i
standard_res_encoder_template = tf.make_template('standard_res_encoder_state_template', standard_res_encoder)



def conv_lstm_encoder(inputs, nr_slices=4, filter_size=2, nr_downsamples=1, nr_residual=2, nonlinearity="concat_elu", gated=True):
  for i in xrange(nr_slices):
    x_conv_compression = standard_res_encode_template(inputs[i], filter_size=filter_size, nr_downsamples=nr_downsamples, nr_residual=nr_residual, nonlinearity=nonlinearity, gated=gated)
    x_conv_
   

def conv_lstm(inputs, hidden_state_1=None, hidden_state_2=None)

def standard_res_decoder(inputs, final_size, filter_size=2, nr_downsamples=5, nr_residual=2, nonlinearity="concat_elu", gated=True):
  y_i = inputs
  nonlinearity = set_nonlinearity(nonlinearity)

  filter_size = filter_size*pow(2,nr_downsamples-2)
  for i in xrange(nr_downsamples-1):
    print(filter_size)
    y_i = transpose_conv_layer(y_i, 4, 2, filter_size, "up_conv_" + str(i))
    for j in xrange(nr_residual):
      y_i = res_block(y_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=1, gated=gated, name="resnet_up_sampled_" + str(i) + "_nr_residual_" + str(j+1))
    filter_size = filter_size/2
  y_i = transpose_conv_layer(y_i, 4, 2, final_size, "up_conv_" + str(nr_downsamples))
  return tf.nn.tanh(y_i)
standard_res_decoder_template = tf.make_template('standard_res_decoder_state_template', standard_res_decoder)


