import tensorflow as tf
import os
import logging
import io
import PIL
import pickle
import math
import numpy as np
from PIL import Image
from scipy import ndimage

image_width = 200
image_height = 200
pixel_depth = 255.0  # Number of levels per pixel.
num_channels = 1 # grayscale
batch_size = 10
batch_repeat = 32
patch_size = 5
depth = 32
last_depth = 50
num_hidden = 128
dropout = 0.8
num_steps = 20001

def load_tooth(folder):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_height, image_width),
                         dtype=np.float32)
  num_images = 0
  test_size = 0.2
  for image in image_files:
    image_file = os.path.join(folder, image)
    # print('image', image_file)
    try:
      image_data = (np.array(PIL.Image.open(image_file).convert('L')) -
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_height, image_width):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:num_images, :, :]
  np.random.shuffle(dataset)
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset[0:int(num_images * (1 - test_size)), :, :], dataset[int(num_images * (1 - test_size)):num_images, :, :]

def maybe_pickle(data_folders, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder
    dataset_names.append(set_filename)
    if os.path.exists(set_filename + 'train.pickle') and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      train_dataset, test_dataset = load_tooth(folder)
      try:
        print('train' + set_filename)
        with open( folder + 'train.pickle', 'wb') as f:
          pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
        with open( folder + 'test.pickle', 'wb') as f:
          pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  return dataset_names

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def confusion_matrix(predictions, labels):
    print(predictions, labels)
    return tf.contrib.metrics.confusion_matrix(np.argmax(predictions, 1), np.argmax(labels, 1))

def merge_datasets(datasets):
    train_d = []
    train_l = []
    test_d = []
    test_l = []
    for index, foldername in enumerate(datasets):
        dataset = pickle.load(open(foldername + 'train.pickle', "rb"))
        train_d.append(dataset)
        train_l.append(np.array([np.eye(len(datasets))[index] for a in dataset]))
        print(len(dataset), np.mean(dataset), np.std(dataset))
        dataset = pickle.load(open(foldername + 'test.pickle', "rb"))
        test_d.append(dataset)
        test_l.append(np.array([np.eye(len(datasets))[index] for a in dataset]))
        print(len(dataset), np.mean(dataset), np.std(dataset))
    train_d = np.expand_dims(np.concatenate(tuple(train_d)), axis=3)
    train_l = np.concatenate(tuple(train_l))
    test_d = np.expand_dims(np.concatenate(tuple(test_d)), axis=3)
    test_l = np.concatenate(tuple(test_l))
    return train_d, train_l, test_d, test_l

train_folders = [
                    'train/11', 'train/12', 'train/13', 'train/14', 'train/15', 'train/16', 'train/17', 'train/18',
                ]

dataset_names = maybe_pickle(train_folders)
train_features, train_labels, test_features, test_labels = merge_datasets(dataset_names)

num_labels = len(train_folders)

graph = tf.Graph()
with graph.as_default():
    # Input data.
    dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(batch_repeat)
    iterator = dataset.make_one_shot_iterator()
    next_element, next_label = iterator.get_next()

    tf_test_dataset = tf.constant(test_features)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [5, 5, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [3, 3, depth, 2 * depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[2 * depth]))
    layer3_weights = tf.Variable(tf.truncated_normal(
        [3, 3, 2 * depth, 2 * depth], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[2 * depth]))
    layer4_weights = tf.Variable(tf.truncated_normal(
        [3, 3, 2 * depth, last_depth], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[last_depth]))
    full_weights = tf.Variable(tf.truncated_normal(
        [int(math.ceil(image_height / 16.0) * math.ceil(image_width / 16.0) * last_depth), num_hidden], stddev=0.1))
    full_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    logits_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    logits_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        print(data)
        # conv1 = inception2d(data, 1, depth)
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + layer1_biases)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + layer2_biases)
        pool2 = tf.nn.max_pool(hidden2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv3 = tf.nn.conv2d(pool2, layer3_weights, [1, 1, 1, 1], padding='SAME')
        hidden3 = tf.nn.relu(conv3 + layer3_biases)
        pool3 = tf.nn.max_pool(hidden3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv4 = tf.nn.conv2d(pool3, layer4_weights, [1, 1, 1, 1], padding='SAME')
        hidden4 = tf.nn.relu(conv4 + layer4_biases)
        pool4 = tf.nn.max_pool(hidden4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        shape = pool4.get_shape().as_list()

        reshape = tf.reshape(pool4, [-1, shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, full_weights) + full_biases)
        hidden = tf.nn.dropout(hidden, dropout)
        return tf.matmul(hidden, logits_weights) + logits_biases

    logits = model(next_element)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=next_label, logits=logits))
    loss_summary = tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.005
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, num_steps, 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    confusion = tf.contrib.metrics.confusion_matrix(tf.argmax(test_prediction, 1), np.argmax(test_labels, 1))
    # confusion_summary = tf.summary.image('confusion', tf.reshape( confusion, [1, num_labels, num_labels, 1]))

    merged = tf.summary.merge([loss_summary])
    merged_all = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/train/')
    saver = tf.train.Saver()

    step = 0
    with tf.train.MonitoredTrainingSession() as sess:
        while not sess.should_stop():
            if (step % 100 == 0):
                _, l, predictions, n_l, t_predictions, summary, c = sess.run([optimizer, loss, train_prediction, next_label, test_prediction, merged_all, confusion])
                print('Minibatch loss at step %d: %f' % (step, l))
                # print((predictions, next_label))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, n_l))
                print('Validation accuracy: %.1f%%' % accuracy(
                        t_predictions, test_labels))
                print('Validation confusion_matrix:')
                print(c)
            else:
                _, l, predictions, summary = sess.run([optimizer, loss, train_prediction, merged])
            train_writer.add_summary(summary, step)
            step += 1

            # if step % 500 == 0:
            #      save_path = saver.save(sess, "model.ckpt")
            #      print("Model saved in path: %s" % save_path)

            if step > num_steps:
                break
