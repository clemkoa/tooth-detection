import numpy as np
import os
import sys
import glob
import tensorflow as tf
from matplotlib import pyplot as plt
import random
from PIL import Image
import json
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def preprocess_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    cl = clahe.apply(img)
    image = Image.fromarray(cl)
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


PATH_TO_FROZEN_GRAPH = '/Users/clementjoudet/Desktop/dev/tooth-detection/models/transfer/inference/frozen_inference_graph.pb'
PATH_TO_LABELS = '/Users/clementjoudet/Desktop/dev/tooth-detection/data/golden_label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = '/Users/clementjoudet/Desktop/perso/tooth-detection/dataset/TAGS MANQUANTS/brazil_cat10'
TEST_IMAGE_PATHS = sorted(glob.glob(PATH_TO_TEST_IMAGES_DIR + '/*.jpg'))
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

vott_output = {}
output_path = PATH_TO_TEST_IMAGES_DIR + '.json'

id = 0
cats = []
for image_path in TEST_IMAGE_PATHS:
    print(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape

    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph)

    detection_boxes = output_dict['detection_boxes']
    detection_boxes[:, 0] = detection_boxes[:, 0] * height
    detection_boxes[:, 2] = detection_boxes[:, 2] * height
    detection_boxes[:, 1] = detection_boxes[:, 1] * width
    detection_boxes[:, 3] = detection_boxes[:, 3] * width
    detection_boxes = detection_boxes.astype(int)
    SCORE_THRESHOLD = 0.5
    vott_output[image_path.split('/')[-1]] = []
    for i in range(len(detection_boxes)):
        if output_dict['detection_scores'][i] > SCORE_THRESHOLD:
            cat = category_index[output_dict['detection_classes'][i]]['name']
            ymin, xmin, ymax, xmax = detection_boxes[i]
            cats.append(cat)
            obj = {
                'x1': xmin,
                'y1': ymin,
                'x2': xmax,
                'y2': ymax,
                'width': width,
                'height': height,
                'box': {
                    'x1': xmin,
                    'y1': ymin,
                    'x2': xmax,
                    'y2': ymax
                },
                "points": [{
                        "x": xmin,
                        "y": ymin
                    }, {
                        "x": xmax,
                        "y": ymin
                    }, {
                        "x": xmax,
                        "y": ymax
                    }, {
                        "x": xmin,
                        "y": ymax
                    }],
                'id': id,
                'type': 'rect',
                'tags': [str(cat)],
            }
            id += 1
            vott_output[image_path.split('/')[-1]].append(obj)


    r = lambda: random.randint(0,255)

    final_obj = {
        'frames': vott_output,
        'framerate': 1,
        'inputTags': ','.join(list(set(cats))),
        'suggestiontype': 'track',
        'scd': False,
        'visitedFrames': list(vott_output.keys())[::-1],
        'tag_colors':['#%02X%02X%02X' % (r(),r(),r()) for j in range(len(list(set(cats))))]
    }

    outfile = open(output_path, 'w')
    json.dump(final_obj, outfile, separators=(',',':'))
    outfile.close()
