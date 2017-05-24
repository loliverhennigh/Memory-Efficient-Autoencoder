
import numpy as np
import matplotlib.pyplot as plt

def make_input_wavy_2d(shape):
  rand_slope = 0.5 * np.random.rand(len(shape))

  axis_0 = np.expand_dims(rand_slope[0] * np.arange(shape[0]), 1)
  ones_0 = np.expand_dims(np.ones(shape[0]), 0)
  axis_1 = np.expand_dims(rand_slope[1] * np.arange(shape[1]), 1)
  ones_1 = np.expand_dims(np.ones(shape[1]), 0)

  wavy_0 = axis_0 * ones_0
  wavy_1 = axis_1 * ones_1
  wavy = np.sin(wavy_0) + 2.0*np.sin(wavy_0) + 6.0*np.sin(wavy_0) + np.sin(np.transpose(wavy_1)) + 3.0*np.sin(np.transpose(wavy_1)) + 10.0*np.sin(np.transpose(wavy_1))
  #wavy = np.sin(wavy)
  return np.expand_dims(wavy,2)/np.max(wavy)

def make_batch(batch_size, shape, input_maker=make_input_wavy_2d):
  wavy_batch = np.zeros([batch_size] + shape + [1])
  for i in xrange(batch_size):
    wavy_batch[i] = input_maker(shape) 
  return wavy_batch


