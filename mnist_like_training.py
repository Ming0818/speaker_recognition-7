# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
# import utils
import tables
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf

FLAGS = None
PLOT_DIR = './out/plots'

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'development_dataset_path', '../../data/development_sample_dataset_speaker.hdf5',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'batch_size', 3, 'The number of samples in each batch. It will be the number of samples distributed for all clones.')


# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS


def deepnn(x, x_size, y_size, class_number):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  # with tf.name_scope('reshape'):
  #   x_image = tf.reshape(x, [-1, x_size, y_size, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 128])
    b_conv1 = bias_variable([128])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    tf.add_to_collection('conv_weights', W_conv1)
    tf.add_to_collection('conv_output', h_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 128, 256])
    b_conv2 = bias_variable([256])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)



  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([10 * 20 * 256, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 10 * 20 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, class_number])
    b_fc2 = bias_variable([class_number])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    tf.add_to_collection('fc2', y_conv)

  
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# def plot_conv_output(conv_img, name):
#     """
#     Makes plots of results of performing convolution
#     :param conv_img: numpy array of rank 4
#     :param name: string, name of convolutional layer
#     :return: nothing, plots are saved on the disk
#     """
#     # make path to output folder
#     plot_dir = os.path.join(PLOT_DIR, 'conv_output')
#     plot_dir = os.path.join(plot_dir, name)

#     # create directory if does not exist, otherwise empty it
#     utils.prepare_dir(plot_dir, empty=True)

#     w_min = np.min(conv_img)
#     w_max = np.max(conv_img)

#     # get number of convolutional filters
#     num_filters = conv_img.shape[3]

#     # get number of grid rows and columns
#     grid_r, grid_c = utils.get_grid_dim(num_filters)

#     # create figure and axes
#     fig, axes = plt.subplots(min([grid_r, grid_c]),
#                              max([grid_r, grid_c]))

#     # iterate filters
#     for l, ax in enumerate(axes.flat):
#         # get a single image
#         img = conv_img[0, :, :,  l]
#         # put it on the grid
#         ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
#         # remove any labels from the axes
#         ax.set_xticks([])
#         ax.set_yticks([])
#     # save figure
#     plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')



# Load the sample artificial dataset
fileh = tables.open_file(FLAGS.development_dataset_path, mode='r')

##################################
######### Check dataset ##########
##################################

# Train
print("Train data shape:", fileh.root.utterance_train.shape)
print("Train label shape:", fileh.root.label_train.shape)

# Test
print("Test data shape:", fileh.root.utterance_test.shape)
print("Test label shape:",fileh.root.label_test.shape)

# Get the number of subjects
num_subjects = len(np.unique(fileh.root.label_train[:]))
print("np.unique(fileh.root.label_train[:]) {0}".format(np.unique(fileh.root.label_train[:])))

label_train = np.zeros((fileh.root.label_train.shape[0],num_subjects))
label_test = np.zeros((fileh.root.label_test.shape[0],num_subjects))

print("Train label shape:",label_train.shape)
print("Test label shape:",label_test.shape)

# create a vector a label from in label value
for idx in range(1,fileh.root.label_train.shape[0]):
  label_train[idx,fileh.root.label_train[idx]] = 1


for idx in range(1,fileh.root.label_test.shape[0]):
  label_test[idx,fileh.root.label_test[idx]] = 1

  
print("Train label shape:",label_train.shape)
print("Test label shape:",label_test)
index = 1
def getNextBatchTrain(index_in, size):
  batch = []
  if (index_in + size) < fileh.root.utterance_train.shape[0] :
    batch.append(fileh.root.utterance_train[index_in:index_in + size,:])
    batch.append(label_train[index_in:index_in + size])
    index_in += size
  else:
    batch.append(fileh.root.utterance_train[index_in:end,:])
    batch.append(label_test[index_in:end])
    index_in += size


  if index_in >= fileh.root.utterance_train.shape[0]:
    index_in = 0
  return batch

def main(_):

  x_size = 80
  y_size = 40
  # Create the model
  x = tf.placeholder(tf.float32, [None, x_size,y_size,1])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, num_subjects])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x, x_size, y_size, num_subjects)



  # speech = tf.placeholder(tf.float32, (1, x_size, y_size, 1))
  # label = tf.placeholder(tf.int32, (num_subjects))
  # batch_dynamic = tf.placeholder(tf.int32, ())
  # margin_imp_tensor = tf.placeholder(tf.float32, ())

 
  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  saver = tf.train.Saver()

# Initializing the variables
  init = tf.initialize_all_variables()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(init)
    for i in range(1000):
      # batch = mnist.train.next_batch(50)
      batch = getNextBatchTrain(index,50)
      if i % 100 == 0:
        # train_accuracy = accuracy.eval(feed_dict={
        #     x: batch[0], y_: batch[1], keep_prob: 1.0})
        # print('step %d, training accuracy %g' % (i, train_accuracy))
        print('test accuracy %g' % accuracy.eval(feed_dict={
        x: fileh.root.utterance_test, y_: label_test, keep_prob: 1.0}))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: fileh.root.utterance_test, y_: label_test, keep_prob: 1.0}))

      # get output of all convolutional layers
    # here we need to provide an input image
    # conv_out = sess.run([tf.get_collection('fc1')], feed_dict={x: fileh.root.utterance_test[:1]})
    # fc2_out = sess.run([tf.get_collection('fc2')], feed_dict={x: fileh.root.utterance_test[:1] , keep_prob: 1.0})
    # print('fc2_out {0}'.format(fc2_out))
    # print('conv_out {0}'.format(conv_out))
    # for i, c in enumerate(conv_out[0]):
    #     plot_conv_output(c, 'conv{}'.format(i))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)