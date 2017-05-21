
import numpy as np
import matplotlib.pyplot as plt

def make_input_wavy_2d(shape):
  rand_slope = 0.2 * np.random.rand(len(shape)) + 0.1

  axis_0 = np.expand_dims(rand_slope[0] * np.arange(shape[0]), 1)
  ones_0 = np.expand_dims(np.ones(shape[0]), 0)
  axis_1 = np.expand_dims(rand_slope[1] * np.arange(shape[1]), 1)
  ones_1 = np.expand_dims(np.ones(shape[1]), 0)

  wavy_0 = axis_0 * ones_0
  wavy_1 = axis_1 * ones_1
  wavy = wavy_1 + np.transpose(wavy_1)
  wavy = np.sin(wavy)
  return wavy

def make_batch(batch_size, shape, input_maker=make_input_wavy_2d):
  wavy_batch = np.zeros([batch_size] + shape)
  for i in xrange(batch_size):
    wavy_batch[i] = input_maker(shape) 
  return wavy_batch


wavy = make_batch(3, [100,100])
print(wavy[0].shape)
plt.imshow(wavy[0])
plt.show()



