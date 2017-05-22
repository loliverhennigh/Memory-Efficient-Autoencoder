
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


def standard_res_decoding(inputs, final_size, filter_size=2, nr_downsamples=5, nr_residual=2, nonlinearity="concat_elu", gated=True):
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


