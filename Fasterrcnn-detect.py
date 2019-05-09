#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from glob import glob
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import collections
from object_detection.utils import ops as utils_ops
import logging
from google.protobuf import text_format
# as python package
from object_detection.protos import string_int_label_map_pb2
%matplotlib inline


# manage categories map file
def _validate_label_map(label_map):
    for item in label_map.item:
        if item.id < 0:
            raise ValueError('Label map ids should be >= 0.')
        if (item.id == 0 and item.name != 'background' and item.display_name != 'background'):
            raise ValueError('Label map id 0 is reserved for the background label')

def create_category_index(categories):
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index



def get_max_label_map_index(label_map):
    return max([item.id for item in label_map.item])

def convert_label_map_to_categories(label_map,
                                    max_num_classes,
                                    use_display_name=True):
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append({'id': class_id + label_id_offset,
          'name': 'category_{}'.format(class_id + label_id_offset)})
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info(
              'Ignore item %d since it falls outside of requested '
              'label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
    return categories


def load_labelmap(path):
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
        text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
        label_map.ParseFromString(label_map_string)
        _validate_label_map(label_map)
    return label_map


def get_label_map_dict(label_map_path,
                       use_display_name=False,
                       fill_in_gaps_and_background=False):
    label_map = load_labelmap(label_map_path)
    label_map_dict = {}
    for item in label_map.item:
        if use_display_name:
            label_map_dict[item.display_name] = item.id
        else:
            label_map_dict[item.name] = item.id
    if fill_in_gaps_and_background:
        values = set(label_map_dict.values())

    if 0 not in values:
        label_map_dict['background'] = 0
    if not all(isinstance(value, int) for value in values):
        raise ValueError('The values in label map must be integers in order to'
                       'fill_in_gaps_and_background.')
    if not all(value >= 0 for value in values):
        raise ValueError('The values in the label map must be positive.')

    if len(values) != max(values) + 1:
        for value in range(1, max(values)):
            if value not in values:
                label_map_dict[str(value)] = value

    return label_map_dict


def create_categories_from_labelmap(label_map_path, use_display_name=True):
    label_map = load_labelmap(label_map_path)
    max_num_classes = max(item.id for item in label_map.item)
    return convert_label_map_to_categories(label_map, max_num_classes, use_display_name)

def create_category_index_from_labelmap(label_map_path, use_display_name=True):
    categories = create_categories_from_labelmap(label_map_path, use_display_name)
    return create_category_index(categories)

def create_class_agnostic_category_index():
    return {1: {'id': 1, 'name': 'object'}}
	
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# draw bounding boxes

def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
    
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = '{}%'.format(int(100*scores[i]))
                    else:
                        display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = 'Orange'

  # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
                draw_mask_on_image_array(
                  image,
                  box_to_instance_masks_map[box],
                  color=color)
        if instance_boundaries is not None:
            draw_mask_on_image_array(
              image,
              box_to_instance_boundaries_map[box],
              color='red',
              alpha=1.0)
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            draw_keypoints_on_image_array(
              image,
              box_to_keypoints_map[box],
              color=color,
              radius=line_thickness / 2,
              use_normalized_coordinates=use_normalized_coordinates)

    return image

def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):

    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))

def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)
        
def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    

    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
                    [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                      text_bottom)],
                    fill=color)
        draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str,
                fill='black',
                font=font)
        text_bottom -= text_height - 2 * margin





# predict func
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
          # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                      'num_detections', 'detection_boxes', 'detection_scores',
                      'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

          # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict




# model restore file path
frozen_model_path = "../input/resnetv2/rennetv2/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28/frozen_inference_graph.pb"
label_map_path = "../input/fasterrcnn/oid_v4_label_map.pbtxt.txt"
test_image_path = glob("../input/google-ai-open-images-object-detection-track/test/challenge2018_test/*.jpg")


# load categories
category_index = create_category_index_from_labelmap(label_map_path, use_display_name=True)
#load labels
classtable = pd.read_csv('../input/classlable/class-boxlable.csv', header=None)
class_dict = {classtable.iloc[i,0]:classtable.iloc[i,1] for i in range(classtable.shape[0])}


# model restore
detect_graph = tf.Graph()
with detect_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(frozen_model_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')



# predict
results = []
for one in test_image_path:
    image = Image.open(one)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detect_graph)
  # Visualization of the results of a detection.
    visualize_boxes_and_labels_on_image_array(
                                              image_np,
                                              output_dict['detection_boxes'],
                                              output_dict['detection_classes'],
                                              output_dict['detection_scores'],
                                              category_index,
                                              instance_masks=output_dict.get('detection_masks'),
                                              use_normalized_coordinates=True,
                                              line_thickness=8)
    
    plt.figure(figsize=(12,8))
    plt.imshow(image_np)
    PredictionString = ''
    for i in range(output_dict['num_detections']):
        res = class_dict[category_index[output_dict['detection_classes'][i]]['name']]
        res += ' ' + str(output_dict['detection_scores'][i])
        for box_locate in range(4):
            res += ' ' + str(output_dict['detection_boxes'][i][box_locate])
        res += '\n'
        PredictionString += res
    results.append({'ImageId':one[78:][:-4] , 'PredictionString': PredictionString.rstrip('\n')})




# create submit file
submit_file = pd.DataFrame.from_records(results, columns = ['ImageId', 'PredictionString'])
submit_file.to_csv('submit.csv', index=None, header=True)



