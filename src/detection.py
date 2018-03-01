#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

#matplotlib inline
#sys.path.append("..")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def region_detection(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES, image, score_thresh=0.5):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            #image = Image.open(PATH_TO_IMAGE)
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict = {image_tensor: image_np_expanded}
                    )
            vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    min_score_thresh=score_thresh,
                    use_normalized_coordinates=True,
                    line_thickness=8
                    )
            #plt.figure(figsize=(12,8))
            #plt.imshow(image_np)
            #im = Image.fromarray(image_np)
            #im.save('../testdata/'+str(cnt+1)+'.jpg')
            if np.squeeze(scores)[0] > score_thresh:
                flag = True
            else:
                flag = False
            #print np.squeeze(boxes), np.squeeze(scores), classes, num_detections
    return  flag, image_np, np.squeeze(boxes), np.squeeze(scores)

if __name__ == '__main__':
    PATH_TO_CKPT = '../../mydata/tfmodels/ssd_inception_v2/train/output_inference_graph.pb'
    PATH_TO_LABELS = '../../mydata/tfdata/pascal_label_map.pbtxt'
    PATH_TO_IMAGE = []
    rootdir = '../../mydata/VOCdevkit/VOC2007/JPEGImages'
    lists = os.listdir(rootdir)
    for i in range(len(lists)):
        path = os.path.join(rootdir, lists[i])
        PATH_TO_IMAGE.append(path)
    NUM_CLASSES = 1
    for image_file in PATH_TO_IMAGE:
        image = Image.open(image_file)
        region_detection(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES, image, 0.5)

