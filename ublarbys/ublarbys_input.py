# Routine for decoding the UBLARBYS tensoflowed data

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 224 x 224. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 224

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 90753-1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000

def read_ublarbys(filename_queue):
  """Reads and parses examples from UBLARBYS data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  # cheap way to make a hash
  class UBLARBYSRecord(object):
    pass
  result = UBLARBYSRecord()

  reader = tf.TFRecordReader()
  result.key, value= reader.read(filename_queue)
  features = tf.parse_single_example(
    value,
    # Defaults are not specified since both keys are required.
    features={
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'depth': tf.FixedLenFeature([], tf.int64)
  })

  image = tf.decode_raw(features['image_raw'], tf.uint8)


  result.label = tf.cast(features['label'], tf.int32)
  result.height = tf.cast(features['height'], tf.int32)
  result.width = tf.cast(features['width'], tf.int32)
  result.depth = tf.cast(features['depth'], tf.int32)
  result.uint8image = tf.reshape(image, tf.pack([224, 224, 3]))  
  result.uint8image.set_shape([224,224,3])
  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    #print(image,label,batch_size,num_preprocess_threads,min_queue_examples + 3 * batch_size,min_queue_examples)
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images) # these are not ``distorted" yet I guess

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  
  # where is this thing
  filenames = ['/stage/vgenty/train.tfrecords']

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_ublarbys(filename_queue)
  #
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.
  # distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

  # Randomly flip the image horizontally.
  # distorted_image = tf.image.random_flip_left_right(distorted_image)
  
  distorted_image = tf.image.random_flip_left_right(reshaped_image)

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  #distorted_image = tf.image.random_brightness(distorted_image,
  #                                             max_delta=63)
  #distorted_image = tf.image.random_contrast(distorted_image,
  #                                           lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  # no mean subtraction in tensorflow, just whitening, fine
  float_image = tf.image.per_image_whitening(distorted_image)
  
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d UBLARBYS images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for UBLARBYS evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = ['/stage/vgenty/train.tfrecords']
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    raise exception("Fuck")
    filenames = ['validate.tfrecords']
    # filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_ublarbys(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  # resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
  #                                                        width, height)

  # Subtract off the mean and divide by the variance of the pixels.
  # must be exact same as train time
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)
