# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:13:04 2019

@author: GS65 8RF
"""

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}

import os
import json
import xml.etree.cElementTree as ET
def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.lower().strip()
        if label not in label_map:
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

voc07_path = os.path.abspath('D:/Resources/SSD-test(PyTorch)/VOCtrainval-2007')
output_folder = 'D:/Resources/temp'
train_images = list()
train_objects = list()
n_objects = 0

with open(os.path.join(voc07_path, 'ImageSets/Main/trainval.txt')) as f:
    ids = f.read().splitlines()
for id in ids:
    objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
    if len(objects) == 0: continue
        
    n_objects += len(objects)
    train_objects.append(objects)
    train_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.jpg'))
        
assert len(train_objects) == len(train_images)
with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
    json.dump(train_images, j)
with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
    json.dump(train_objects, j)
with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
    json.dump(label_map, j)  # save label map too
print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
    len(train_images), n_objects, os.path.abspath(output_folder)))

