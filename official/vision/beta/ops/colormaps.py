""" Colormaps for visualising segmentation masks """

import tensorflow as tf


def get_colormap(cmap_type='cityscapes', 
                 ignore_label=255):
  """
  Obtain colormap that converts invalid labels outside of its 
  range to white.

  Args:
  cmap_type: type of colormap, currently supports 'cityscapes' only
  ignore_label: label used for parts of the features ignored
  """

  if cmap_type == 'cityscapes':
    cmap = tf.constant([
      [128, 64, 128],
      [244, 35, 232],
      [70, 70, 70],
      [102, 102, 156],
      [190, 153, 153],
      [153, 153, 153],
      [250, 170, 30],
      [220, 220, 0],
      [107, 142, 35],
      [152, 251, 152],
      [0, 130, 180],
      [220, 20, 60],
      [255, 0, 0],
      [0, 0, 142],
      [0, 0, 70],
      [0, 60, 100],
      [0, 80, 100],
      [0, 0, 230],
      [119, 11, 32]
    ], dtype=tf.uint8)

    max_label = cmap.shape[0] - 1
    empty_labels = tf.tile(tf.constant([[255]], dtype=tf.uint8), 
                           multiples=(ignore_label-max_label, 3))
    cmap = tf.concat((cmap, empty_labels), axis=0)
  
  elif cmap_type == 'cityscapes_int':
    # two complement of above values + BGR instead of RGB
    cmap = tf.constant([
      -25149312, 
      -18340876, 
      -28948922, 
      -23304602, 
      -23488066, 
      -23488103, 
      -31544582, 
      -33497892, 
      -31224213, 
      -23528552, 
      -21724672, 
      -29616932, 
      -33554177, 
      -24248320, 
      -28966912, 
      -26985472, 
      -26980352, 
      -18481152, 
      -31454345], dtype=tf.int32)
  else:
    raise ValueError('Invalid colormap type specified %s.' %cmap_type)

  return cmap
