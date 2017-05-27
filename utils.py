
import numpy as np

def image_to_grid(tensor, shape, nr_split):

  # possibly add batch to image
  if len(tensor.shape) < 4:
    tensor = tensor.reshape([1] + shape + [1]) # add zero

  # split in 1 direction 
  tensor_x_split = np.split(tensor, nr_split, 1)
  tensor_split = []
  for i in xrange(nr_split):
    tensor_y_split = np.split(tensor_x_split[i], nr_split, 2)
    for j in xrange(nr_split):
      tensor_split.append(np.expand_dims(tensor_y_split[j], axis=1))

  new_tensor = np.concatenate(tensor_split, 1)
  return new_tensor

def grid_to_image(tensor, shape, nr_split):

  # possibly add batch to image
  if len(tensor.shape) < 4:
    tensor = tensor.reshape([1] + shape + [3*nr_split*nr_split]) # add zero

  tensor = np.split(tensor, nr_split*nr_split, 3)

  # concat in 2 direction
  tensor_y_concat = []
  for i in xrange(nr_split):
    tensor_y_concat.append(np.concatenate(tensor[i*nr_split:(i+1)*nr_split], 2))
  
  new_tensor = np.concatenate(tensor_y_concat, 1)
  print(new_tensor.shape)
  return new_tensor




