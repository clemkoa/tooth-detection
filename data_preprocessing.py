import hashlib
import tensorflow as tf
import os
from lxml import etree
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_folder', 'data', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
FLAGS = flags.FLAGS

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl = clahe.apply(img)
    return cv2.imencode('.jpeg', cl)[1].tostring()

def get_image_full_path(dataset_directory, image_subdirectory, filename):
    full_path = os.path.join(dataset_directory, image_subdirectory, filename + '.png')
    if not os.path.isfile(full_path):
        full_path = os.path.join(dataset_directory, image_subdirectory, filename + '.jpg')
    return full_path

def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       categories,
                       image_subdirectory='JPEGImages'):

    full_path = get_image_full_path(dataset_directory, image_subdirectory, data['filename'])

    encoded_jpg = preprocess_image(full_path)

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
            if obj['name'] in categories:
                xmin.append(float(obj['bndbox']['xmin']) / width)
                ymin.append(float(obj['bndbox']['ymin']) / height)
                xmax.append(float(obj['bndbox']['xmax']) / width)
                ymax.append(float(obj['bndbox']['ymax']) / height)
                classes_text.append(obj['name'].encode('utf8'))
                classes.append(label_map_dict[obj['name']])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
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

def extract_examples_list(dataset, categories, data_dir):
    examples_list = []
    for category in categories:
        examples_path = os.path.join(data_dir, 'ImageSets', 'Main', str(category) + '_' + FLAGS.set + '.txt')
        examples_list += dataset_util.read_examples_list(examples_path)

    return list(set([x for x in examples_list if x]))

def extract_dataset(writer, dataset):
    data_dir = os.path.join(flags.FLAGS.data_folder, dataset)
    annotations_dir = os.path.join(data_dir, 'Annotations')
    label_map_dict = label_map_util.get_label_map_dict(os.path.join(data_dir, 'pascal_label_map.pbtxt'))
    categories = list(label_map_dict.keys())
    examples_list = extract_examples_list(dataset, categories, data_dir)
    print('examples_list', examples_list)
    for idx, example in enumerate(examples_list):
        if idx % 10 == 0:
            print('On image ', idx, ' of ', len(examples_list))
        path = os.path.join(annotations_dir, example + '.xml')
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        tf_example = dict_to_tf_example(data, data_dir, label_map_dict, categories)
        writer.write(tf_example.SerializeToString())

def main(_):
    output_path = os.path.join(flags.FLAGS.data_folder, flags.FLAGS.set + '.record')
    writer = tf.python_io.TFRecordWriter(output_path)
    datasets = [f for f in os.listdir(flags.FLAGS.data_folder) if os.path.isdir(os.path.join(flags.FLAGS.data_folder, f))]
    for dataset in datasets:
        extract_dataset(writer, dataset)
    writer.close()

if __name__ == '__main__':
  tf.app.run()
