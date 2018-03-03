import tensorflow as tf
import os
import logging
import io
import PIL
import pickle
import imageio
import numpy as np
from PIL import Image
from scipy import ndimage

image_width = 50
image_height = 100
pixel_depth = 255.0  # Number of levels per pixel.
num_channels = 1 # grayscale
batch_size = 4
patch_size = 5
depth = 16
num_hidden = 64

def load_tooth(folder):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_height, image_width),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    print('image', image_file)
    try:
      image_data = (np.array(PIL.Image.open(image_file).convert('L')) -
                    pixel_depth / 2) / pixel_depth
      # arr = numpy.array(img)
      # image_data = (imageio.imread(image_file).astype(float) -
      #               pixel_depth / 2) / pixel_depth
      # print(image_data)
      if image_data.shape != (image_height, image_width):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:num_images, :, :]

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

def maybe_pickle(data_folders, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_tooth(folder)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  return dataset_names
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])
def merge_datasets(datasets):
    train_d = []
    train_l = []
    for index, foldername in enumerate(datasets):
        dataset = pickle.load(open(foldername, "rb"))
        train_d.append(dataset)
        train_l.append(np.array([np.eye(2)[index] for a in dataset]))
        print(len(dataset), np.mean(dataset), np.std(dataset))
    # print('train_l', train_l)
    results = np.expand_dims(np.concatenate(tuple(train_d)), axis=3)
    # print('shape', results.shape)
    labels = np.concatenate(tuple(train_l))
    return results, labels

train_folders = ['train/6max', 'train/6mand']
test_folders = ['test/6max', 'test/6mand']
train_datasets = maybe_pickle(train_folders)
test_datasets = maybe_pickle(test_folders)

train_features, train_labels = merge_datasets(train_datasets)

test_features, test_labels = merge_datasets(test_datasets)
num_labels = 2

graph = tf.Graph()
with graph.as_default():
    # Input data.
    dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(20)
    iterator = dataset.make_one_shot_iterator()
    next_element, next_label = iterator.get_next()

    tf_test_dataset = tf.constant(test_features)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_width // 4 * image_height // 4 * depth + 400, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [-1, shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(next_element)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=next_label, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    # valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


    num_steps = 10001
    step = 0
    with tf.train.MonitoredTrainingSession() as sess:
        while not sess.should_stop():
            if (step % 100 == 0):
                _, l, predictions, n_l, t_predictions = sess.run([optimizer, loss, train_prediction, next_label, test_prediction])
                print('Minibatch loss at step %d: %f' % (step, l))
                # print((predictions, next_label))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, n_l))
                print('Validation accuracy: %.1f%%' % accuracy(
                        t_predictions, test_labels))
            else:
                _, l, predictions = sess.run([optimizer, loss, train_prediction])
            step += 1
            if step > num_steps:
                break
