import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
import random
from PIL import Image
import json
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
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

PATH_TO_FROZEN_GRAPH = '/Users/clementjoudet/Desktop/dev/tooth-detection/models/index/cloud/inference/frozen_inference_graph.pb'
PATH_TO_LABELS = '/Users/clementjoudet/Desktop/dev/tooth-detection/data/pascal_label_map_index.pbtxt'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

PATH_TO_TEST_IMAGES_DIR = '/Users/clementjoudet/Desktop/dev/tooth-detection/data/test/JPEGImages'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.png'.format(i)) for i in range(1, 10) ]

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
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
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

vott_output = {}
output_path = 'data/test/JPEGImages.json'

id = 0
for image_path in TEST_IMAGE_PATHS:
    # image = Image.open(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    detection_boxes = output_dict['detection_boxes']
    detection_boxes[:, 0] = detection_boxes[:, 0] * height
    detection_boxes[:, 2] = detection_boxes[:, 2] * height
    detection_boxes[:, 1] = detection_boxes[:, 1] * width
    detection_boxes[:, 3] = detection_boxes[:, 3] * width
    detection_boxes = detection_boxes.astype(int)
    SCORE_THRESHOLD = 0.5
    vott_output[image_path.split('/')[-1]] = []
    cats = []
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
                'id': id,
                'type': 'rect',
                'tags': [cat],
            }
            id += 1
            vott_output[image_path.split('/')[-1]].append(obj)


    r = lambda: random.randint(0,255)

    final_obj = {
        'frames': vott_output,
        'framerate': 1,
        'inputTags': ','.join(list(set(cats))),
        'suggestiontype':'track',
        'scd': False,
        'visitedFrames': list(vott_output.keys())[::-1],
        'tag_colors':['#%02X%02X%02X' % (r(),r(),r()) for j in range(len(list(set(cats))))]
    }

    outfile = open(output_path, 'w')
    json.dump(final_obj, outfile, separators=(',',':'), sort_keys=True)
    outfile.close()
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     output_dict['detection_boxes'],
    #     output_dict['detection_classes'],
    #     output_dict['detection_scores'],
    #     category_index,
    #     instance_masks=output_dict.get('detection_masks'),
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    # im = Image.fromarray(image_np)
    # im.show()
