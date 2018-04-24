# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import tensorflow as tf
import os
import logging
import io
from lxml import etree
import PIL.Image
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import dataset_util

r"""Convert raw PASCAL dataset type to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --output_path=/home/user/pascal.record
"""


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', os.path.join('data', flags.FLAGS.set + '.record'), 'Path to output TFRecord')
SETS = ['train', 'val', 'trainval', 'test']
FLAGS = flags.FLAGS

label_golden = { 'implant': 1, 'endo': 2, 'restauration': 3, 'racine': 4 }
selected = ['implant', 'endo', 'restauration', 'racine']

def create_directory_if_not_exists(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)

def save_cropped_images(data,
                        dataset_directory,
                        label_map_dict,
                        image_subdirectory = 'JPEGImages',
                        output_directory = FLAGS.set,
                        categories = [],
                        dataset_name = 'custom'):

  # output_dir_dict = {16: '6max', 26: '6max', 36: '6mand', 46: '6mand'}
  output_directory = 'train'
  full_path = os.path.join(dataset_directory, image_subdirectory, data['filename'] + '.png')
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  for i, obj in enumerate(data['object']):
    if obj['name'] in ['Implant', 'implant', 'endo', 'Endo', 'restauration', 'Restauration', 'racine', 'Racine']:
        continue
    if int(obj['name']) not in categories:
        continue
    xmin = float(obj['bndbox']['xmin'])
    ymin = float(obj['bndbox']['ymin'])
    xmax = float(obj['bndbox']['xmax'])
    ymax = float(obj['bndbox']['ymax'])
    path = os.path.join(output_directory, str(obj['name']), dataset_name + str(data['filename']) + '-' + str(i) + '.png')
    print(path)
    image.crop((xmin, ymin, xmax, ymax)).resize((100, 200)).convert('L').save(path)
    flipped_path = os.path.join(output_directory, str(get_opposite_category(int(obj['name']))), dataset_name + str(data['filename']) + '-' + str(i) + 'flipped.png')
    print('flipping to :', flipped_path)
    image.crop((xmin, ymin, xmax, ymax)).resize((100, 200)).transpose(Image.FLIP_LEFT_RIGHT).convert('L').save(flipped_path)

def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  full_path = os.path.join(dataset_directory, image_subdirectory, data['filename'] + '.png')
  # print('full_path', full_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG': #TODO be sure that all PNG work with this
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  if 'object' not in data.keys():
      print('No label detected in the xml format')
      return
  for obj in data['object']:
      if obj['name'] in selected:
            # print(obj, label_map_dict[obj['name']])
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_golden[obj['name']])

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example

def get_opposite_category(i):
    if (i - 10 % 20) // 10 in [1,3]:
        return i - 10
    else:
        return i + 10

def generate_cropped_images():
    datasets = ['rothschild', 'gonesse', 'google-image', 'noor']
    # datasets = ['google-image', 'noor']
    categories = [11, 12, 13, 14, 15, 16, 17, 18,
                21, 22, 23, 24, 25, 26, 27, 28,
                31, 32, 33, 34, 35, 36, 37, 38,
                41, 42, 43, 44, 45, 46, 47, 48]

    output_directory = FLAGS.set
    for category in categories:
          create_directory_if_not_exists(output_directory)
          create_directory_if_not_exists(os.path.join(output_directory, str(category)))

    for dataset in datasets:
          data_dir = os.path.join('data', dataset)
          examples_path = os.path.join(data_dir, 'ImageSets', 'Main', str(categories[0]) + '_' + FLAGS.set + '.txt')
          label_map_dict = label_map_util.get_label_map_dict(os.path.join(data_dir, 'pascal_label_map.pbtxt'))
          print('label_map_dict', label_map_dict)
          annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
          examples_list = dataset_util.read_examples_list(examples_path)
          examples_list = [x for x in examples_list if x]
          print(examples_list)

          for idx, example in enumerate(examples_list):
                path = os.path.join(annotations_dir, example + '.xml')
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                if 'object' not in data.keys():
                    print('No label, ignoring ', path)
                    continue
                save_cropped_images(data, data_dir, label_map_dict, categories=categories, dataset_name = dataset)

def main(_):
  # generate_cropped_images()

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    datasets = ['gonesse','rothschild', 'google-image', 'noor']
    categories = selected
    for dataset in datasets:
        examples_list = []
        data_dir = os.path.join('data', dataset)
        print('data_dir', data_dir)
        annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
        label_map_dict = label_map_util.get_label_map_dict(os.path.join(data_dir, 'pascal_label_map.pbtxt'))
        for category in categories:
            examples_path = os.path.join(data_dir, 'ImageSets', 'Main', str(category) + '_' + FLAGS.set + '.txt')
            examples_list += dataset_util.read_examples_list(examples_path)

        examples_list = list(set([x for x in examples_list if x]))
        print('examples_list', examples_list)
        for idx, example in enumerate(examples_list):
            if idx % 10 == 0:
                print('On image ', idx, ' of ', len(examples_list))
            path = os.path.join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
            if 'object' not in data.keys():
                print('No label, ignoring ', path)
                continue

            tf_example = dict_to_tf_example(data, data_dir, label_map_dict)
            writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
  tf.app.run()
