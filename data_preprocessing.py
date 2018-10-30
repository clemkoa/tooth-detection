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

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', os.path.join('data', flags.FLAGS.set + '.record'), 'Path to output TFRecord')
SETS = ['train', 'val', 'trainval', 'test']
FLAGS = flags.FLAGS

label_golden = { 'root': 1, 'implant': 2, 'restoration': 3, 'endodontic': 4 }
selected = ['root', 'implant', 'restoration', 'endodontic']

def create_directory_if_not_exists(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       image_subdirectory='JPEGImages'):

  full_path = os.path.join(dataset_directory, image_subdirectory, data['filename'] + '.png')
  if not os.path.isfile(full_path):
      full_path = os.path.join(dataset_directory, image_subdirectory, data['filename'] + '.jpg')
  # print('full_path', full_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format not in ['JPEG', 'PNG']:
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
  else:
      for obj in data['object']:
          if obj['name'] in selected:
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

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    datasets = ['gonesse102', 'gonesse67', 'gonesse97', 'rothschild', 'google', 'noor', 'ufba']
    categories = selected
    for dataset in datasets:
        print(dataset)
        examples_list = []
        data_dir = os.path.join('data', dataset)
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

            tf_example = dict_to_tf_example(data, data_dir, label_map_dict)
            writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
  tf.app.run()
