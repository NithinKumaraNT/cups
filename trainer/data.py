#!/usr/bin/env python
"""This file contains the method that creates data and labels from a directory"""

import os
import cv2
import numpy as np


def create_data_with_labels(image_dir):
    """Gets numpy data and label array from images that are in the folders that are
    in the folder which was given as a parameter. The folders that are in that folder
    are identified by the mug they represent and the folder name starts with the label."""
    mug_dirs = [f for f in os.listdir(image_dir) if not f.startswith('.')]
    mug_files = []

    for mug_dir in mug_dirs:
        mug_image_files = [image_dir + mug_dir + '/' + '{0}'.format(f)
                           for f in os.listdir(image_dir + mug_dir) if not f.startswith('.')]
        mug_files += [mug_image_files]

    num_images = len(mug_files[0])
    images_np_arr = np.empty([len(mug_files), num_images, 64, 64, 3], dtype=np.float32)

    for mug, _ in enumerate(mug_files):
        for mug_image in range(num_images):
            img = cv2.imread(mug_files[mug][mug_image])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            images_np_arr[mug][mug_image] = img / 255.

    data = images_np_arr[0]
    labels = np.full(num_images, int(mug_dirs[0][0]))

    for i in range(1, len(mug_dirs)):
        data = np.append(data, images_np_arr[i], axis=0)
        labels = np.append(labels, np.full(num_images, int(mug_dirs[i][0])), axis=0)

    return data, labels
