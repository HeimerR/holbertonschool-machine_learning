#!/usr/bin/env python3
"""Load Images """
import glob
import numpy as np
import csv
import cv2


def load_images(images_path, as_array=True):
    """
        images_path is the path to a directory from
                which to load images
        as_array is a boolean indicating whether
                the images should be loaded as one numpy.ndarray
                If True, the images should be loaded as a numpy.ndarray
                        of shape (m, h, w, c) where:
                m is the number of images
                h, w, and c are the height, width, and number of channels
                        of all images, respectively
                If False, the images should be loaded as a list of1
                individual numpy.ndarrays

        All images should be loaded in RGB format
        The images should be loaded in alphabetical order by filename
        Returns: images, filenames
        images is either a list/numpy.ndarray of all images
        filenames is a list of the filenames associated with each image
            in images
    """
    image_paths = glob.glob(images_path + "/*")
    images_names = [path.split('/')[-1] for path in image_paths]
    idx = np.argsort(images_names)
    images = [cv2.imread(img) for img in image_paths]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    images = images[idx]  # sorting
    filenames = images_names[idx]  # sorting
    if as_array:
        images = np.concatenate(images)

    return images, filenames


def load_csv(csv_path, params={}):
    """ that loads the contents of a csv file as a list of lists:

    csv_path is the path to the csv to load
    params are the parameters to load the csv with
    Returns: a list of lists representing the contents found in csv_path
    """
    csv_list = []
    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, params)
        for row in csv_reader:
            csv_list.append(row)

    return csv_list
