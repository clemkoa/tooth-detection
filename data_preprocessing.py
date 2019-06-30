import hashlib
import tensorflow as tf
import os
import glob
import cv2
import random
from lxml import etree
from PIL import Image
from shutil import copyfile

from object_detection.utils import label_map_util
from object_detection.utils import dataset_util


RANDOM_SEED = 42
flags = tf.app.flags
flags.DEFINE_string('data_folder', 'data', 'Root directory to raw PASCAL VOC datasets.')
flags.DEFINE_float('train_eval_ratio', 0.8, 'Ratio of training examples from all examples.')
FLAGS = flags.FLAGS

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_image(image_path, horizontal_flip=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl = clahe.apply(img)
    if horizontal_flip:
        cl = cv2.flip(cl, 1)
    # Image.fromarray(cl).show()
    return cv2.imencode('.jpeg', cl)[1].tostring()

def get_image_full_path(dataset_directory, image_subdirectory, filename):
    full_path = os.path.join(dataset_directory, image_subdirectory, filename + '.png')
    if not os.path.isfile(full_path):
        full_path = os.path.join(dataset_directory, image_subdirectory, filename + '.jpg')
    return full_path

def get_horizontal_flipped_index(i):
    # Input: the index number
    # Output: the corresponding horizontally flipped index
    # Example: get_horizontal_flipped_index(13) = 23
    if (i - 10 % 20) // 10 in [1,3]:
        return i - 10
    else:
        return i + 10

def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       categories,
                       image_subdirectory='JPEGImages',
                       flip=False,
                       eval=False):

    full_path = get_image_full_path(dataset_directory, image_subdirectory, data['filename'])
    if eval:
        cop = 'data/inference/' + dataset_directory.split('/')[-2] + '-' + data['filename'] + '.' + full_path.split('.')[-1]
        copyfile(full_path, cop)
    encoded_jpg = preprocess_image(full_path, horizontal_flip=flip)

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
                if flip:
                    print('flip')
                    c = str(get_horizontal_flipped_index(int(obj['name'])))
                    xmin.append(1.0 - float(obj['bndbox']['xmax']) / width)
                    ymin.append(float(obj['bndbox']['ymin']) / height)
                    xmax.append(1.0 - float(obj['bndbox']['xmin']) / width)
                    ymax.append(float(obj['bndbox']['ymax']) / height)
                    classes_text.append(c.encode('utf8'))
                    classes.append(label_map_dict[c])
                else:
                    c = obj['name']
                    xmin.append(float(obj['bndbox']['xmin']) / width)
                    ymin.append(float(obj['bndbox']['ymin']) / height)
                    xmax.append(float(obj['bndbox']['xmax']) / width)
                    ymax.append(float(obj['bndbox']['ymax']) / height)
                    if max([float(obj['bndbox']['xmin']) / width, float(obj['bndbox']['ymin']) / height, float(obj['bndbox']['xmax']) / width, float(obj['bndbox']['ymax']) / height]) > 1.0:
                        print('error')
                        raise Exception('oops')
                    classes_text.append(c.encode('utf8'))
                    classes.append(label_map_dict[c])

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
    all_examples = [p.split('/')[-1].split('.xml')[0] for p in glob.glob(os.path.join(data_dir, 'Annotations', '*.xml'))]
    return all_examples

def get_data(example, annotations_dir):
    path = os.path.join(annotations_dir, example + '.xml')
    with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    return dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

def extract_datapoint(example, annotations_dir, data_dir, label_map_dict, categories, eval=False):
    data = get_data(example, annotations_dir)
    tf_example = dict_to_tf_example(data, data_dir, label_map_dict, categories, eval=eval)
    return tf_example

def extract_flipped_datapoint(example, annotations_dir, data_dir, label_map_dict, categories):
    data = get_data(example, annotations_dir)
    tf_example = dict_to_tf_example(data, data_dir, label_map_dict, categories, flip=True)
    return tf_example

def extract_dataset(dataset):
    data_dir = dataset
    annotations_dir = os.path.join(data_dir, 'Annotations')
    label_map_dict = label_map_util.get_label_map_dict(os.path.join(data_dir, 'pascal_label_map.pbtxt'))
    categories = list(label_map_dict.keys())
    all_examples_list = extract_examples_list(dataset, categories, data_dir)
    train_datapoints = []
    eval_datapoints = []
    random.seed(RANDOM_SEED)
    random.shuffle(all_examples_list)
    limit = int(len(all_examples_list) * flags.FLAGS.train_eval_ratio)
    training_examples = all_examples_list[:limit]
    eval_examples = all_examples_list[limit:]
    for idx, example in enumerate(all_examples_list):
        print(data_dir, example)
        if example in training_examples:
            print('Training')
            train_datapoints.append(extract_datapoint(example, annotations_dir, data_dir, label_map_dict, categories))
        if example in eval_examples:
            print('Eval')
            eval_datapoints.append(extract_datapoint(example, annotations_dir, data_dir, label_map_dict, categories, eval=True))
    return train_datapoints, eval_datapoints

def extract_all_datasets(datasets):
    training = []
    eval = []
    for dataset in datasets:
        print(dataset)
        t, e = extract_dataset(dataset)
        training += t
        eval += e
    return training, eval

def main(_):
    match = '_output'
    training_path = os.path.join(flags.FLAGS.data_folder, 'train' + match + '.record')
    val_path = os.path.join(flags.FLAGS.data_folder, 'val' + match + '.record')
    train_writer = tf.python_io.TFRecordWriter(training_path)
    val_writer = tf.python_io.TFRecordWriter(val_path)
    datasets = glob.glob(flags.FLAGS.data_folder + '/*' + match + '/')
    i = 0
    training, eval = extract_all_datasets(datasets)
    print('Training examples', len(training))
    print('Eval examples', len(eval))
    for tf_example in training:
        train_writer.write(tf_example.SerializeToString())
    train_writer.close()
    for tf_example in eval:
        val_writer.write(tf_example.SerializeToString())
    val_writer.close()

if __name__ == '__main__':
  tf.app.run()
