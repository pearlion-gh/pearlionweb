import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2

#Import the object detection module.
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#Patches:
utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
from flask import Flask, request, Response, jsonify, send_from_directory, abort



def get_image(image_path , img_name):

    model_dir = "saved_model/"
    PATH_TO_LABELS = 'labelmap.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    detection_model = tf.saved_model.load(str(model_dir))
    output_path = './static/detections/'



    #reading the images & apply detection with loaded weight file
    image_np = np.array(Image.open(image_path))
    image_np = np.asarray(image_np)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    filename = img_name
    model_fn = detection_model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    print('detections:')
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,

        use_normalized_coordinates=True,
        line_thickness=8)



    cv2.imwrite(output_path + '{}'.format(filename), cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR))
    print('output saved to: {}'.format(output_path +  '{}'.format(filename)))
