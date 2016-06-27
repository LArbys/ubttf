# I'm going to copy CIFAR and see what's up

# Builds the AlexNet network for UB demo 1 data.
# Current status is that it's fucked

"""

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
#this annoys me but we keep it. print\s"" must be replaced with print(""
from __future__ import print_function 


import os
import re
import sys

# no need...
# import gzip
# import tarfile
# from six.moves import urllib

import tensorflow as tf

import ublarbys_input as ublarbys_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/ublarbys_data',
                           """Path to the UBLARBYS data directory.""")

# Global constants describing the UBLARBYS data set.
IMAGE_SIZE = ublarbys_input.IMAGE_SIZE
NUM_CLASSES = ublarbys_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = ublarbys_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = ublarbys_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

# was 0.1 for CIFAR
INITIAL_LEARNING_RATE = 0.01      # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.

  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def distorted_inputs():
  """Construct distorted input for training using the Reader options.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """

  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  
  data_dir = None
  return ublarbys_input.distorted_inputs(data_dir=data_dir,
                                         batch_size=FLAGS.batch_size)


def inputs(eval_data):

  """Construct input for UBLARBYS evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """

  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  #data_dir = os.path.join(FLAGS.data_dir, 'ublarbys-batches-bin')

  data_dir = None
  return ublarbys_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)


def inference(images):
  """Build the UBLARBYS model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  parameters = []

  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #

  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    _activation_summary(conv1)
    
  pool1 = tf.nn.max_pool(conv1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
  _activation_summary(pool1)

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    _activation_summary(conv2)


  # pool2
  pool2 = tf.nn.max_pool(conv2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
  _activation_summary(pool2)


  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    _activation_summary(conv3)

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    _activation_summary(conv4)

  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    _activation_summary(conv5)

  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')
  _activation_summary(pool5)
  
  
  ####INCLUDE DROPOUT!####
  with tf.variable_scope('fc6') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    print("\tis Dim is fucked?")
    reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])    
    dim = reshape.get_shape()[1].value
    print("\tDim is %s"%dim)

    weights = tf.Variable(tf.truncated_normal(shape=[dim, 4096], 
                                              dtype=tf.float32,
                                              stddev=1e-4),
                          trainable=True,name='weights')

    biases = tf.Variable(tf.constant(0.0, 
                                     shape=[4096], 
                                     dtype=tf.float32),
                         trainable=True, name='biases')

    fc6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc6)

  ####INCLUDE DROPOUT!####
  with tf.variable_scope('fc7') as scope:
    weights = tf.Variable(tf.truncated_normal(shape=[4096, 4096], 
                                              dtype=tf.float32,
                                              stddev=1e-4),
                          trainable=True,name='weights')

    biases = tf.Variable(tf.constant(0.0,
                                     shape=[4096],
                                     dtype=tf.float32),
                         trainable=True, name='biases')

    fc7 = tf.nn.relu(tf.matmul(fc6, weights) + biases, name=scope.name)
    _activation_summary(fc7)
  
  
  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.Variable(tf.truncated_normal(shape=[4096, NUM_CLASSES], 
                                              dtype=tf.float32,
                                              stddev=1e-4),
                          trainable=True,name='weights')

    biases = tf.Variable(tf.constant(0.0, 
                                     shape=[NUM_CLASSES],
                                     dtype=tf.float32),
                         trainable=True, name='biases')
    softmax_linear = tf.add(tf.matmul(fc7, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int32) #why make them 64 bit int? is 32 bits fine?
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in UBLARBYS model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train UBLARBYS model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """

  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # # Decay the learning rate exponentially based on the number of steps.
  # lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
  #                                 global_step,
  #                                 decay_steps,
  #                                 LEARNING_RATE_DECAY_FACTOR,
  #                                 staircase=True)

  lr = INITIAL_LEARNING_RATE # don't decay it since I don't yet understand the function call
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients -- sort of awkward. I could just compute gradients for fun and not apply then ok great
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables -- tensorboard
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients -- also tensorboard... looks like you just specify a dynamic url location
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
