r"""Vision models run inference utility function."""

import os
import glob

import numpy as np
import tensorflow as tf

from official.vision.beta.ops import preprocess_ops

CITYSCAPES_COLORMAP = np.array([
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
    [119, 11, 32],
    [255, 255, 255]
], dtype=np.uint8)


def run_inference(image_path_glob,
                  output_dir,
                  inference_fn,
                  visualise,
                  stitch_original):
  """Runs inference graph for the model, for given directory of images
  
  Args:
    image_path_glob: glob to retrieve image files
    output_dir: directory to output inference results to
    inference_fn: takes and outputs Tensor of shape [batch_size, None, None, 3]
    visualise: flag to use colormap
    stitch_original: flag to stitch original image by the side
  """

  img_filenames = [f for f in glob.glob(image_path_glob, recursive=True)]
  image_dir = image_path_glob.split("*")[0].strip(os.sep).strip('/')

  for img_filename in img_filenames:
    image = tf.io.read_file(img_filename)
    image_format = os.path.splitext(img_filename)[-1]
    if image_format == ".png":
        image = tf.image.decode_png(image)
    elif image_format == ".jpg":
        image = tf.image.decode_jpeg(image)
    else:
        raise NotImplementedError("Unable to decode %s file type." %(image_format))
    
    image = tf.expand_dims(image, axis=0)
    logits = inference_fn(image)
    if not isinstance(logits, np.ndarray):
      logits = logits.numpy()
    logits = np.squeeze(logits)
    if logits.ndim > 2:
        logits = np.argmax(logits, axis=-1).astype(np.uint8)
    seg_map = logits

    if visualise:
      seg_map = CITYSCAPES_COLORMAP[seg_map]
    if stitch_original:
      image = tf.image.resize(image, seg_map.shape[:2])
      image = np.squeeze(image.numpy()).astype(np.uint8)
      seg_map = np.hstack((image, seg_map))
    
    encoded_seg_map = tf.image.encode_png(seg_map)
    save_path = img_filename.replace(image_dir, output_dir)
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    
    tf.io.write_file(save_path, encoded_seg_map)
    print("Visualised %s, saving result at %s" %(img_filename, save_path))
