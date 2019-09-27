import numpy as np
import os
import sys
import glob
import tensorflow as tf
import random
import json
import cv2
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

dir_path = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags
flags.DEFINE_string('PATH_TO_FROZEN_GRAPH', os.path.join(dir_path, 'models', 'output_raw', 'inference', 'frozen_inference_graph.pb'), 'Path to frozen graph')
flags.DEFINE_string('PATH_TO_LABELS', os.path.join(dir_path, 'data', 'golden_label_map.pbtxt'), 'Path to label map')
flags.DEFINE_string('PATH_TO_TEST_IMAGES_DIR', os.path.join(dir_path, 'data', 'inference'), 'Path to image folder')
flags.DEFINE_boolean('no_preprocess', False, '')
FLAGS = flags.FLAGS

def preprocess_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl = clahe.apply(img)
    image = Image.fromarray(cl)
    if flags.FLAGS.no_preprocess:
        print('no preprocessing')
        image = Image.fromarray(img)
    im = Image.new('RGB', image.size)
    im.paste(image)
    return im

def load_image_into_numpy_array(image):
    image = preprocess_image(image)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
  return output_dict

if __name__ == "__main__":
    category_index = label_map_util.create_category_index_from_labelmap(FLAGS.PATH_TO_LABELS, use_display_name=True)
    TEST_IMAGE_PATHS = glob.glob(os.path.join(FLAGS.PATH_TO_TEST_IMAGES_DIR, '*gonesse*.png'))
    OUTPUT_FOLDER = os.path.join('/'.join(FLAGS.PATH_TO_FROZEN_GRAPH.split('/')[:-2]), 'output')
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(FLAGS.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    for image_path in TEST_IMAGE_PATHS:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=50,
            line_thickness=4)
        im = Image.fromarray(image_np)
        print('Saving image to path', os.path.join(OUTPUT_FOLDER, image_path.split('/')[-1]))
        im.save(os.path.join(OUTPUT_FOLDER, image_path.split('/')[-1]))
