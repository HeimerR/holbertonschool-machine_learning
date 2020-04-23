#!/usa/bin/env python3
"""Load Images """
import glob
import numpy as np
import csv
import cv2
import os


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
    images_prev = [cv2.imread(img) for img in image_paths]
    images_prev = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_prev]
    images = []
    filenames = []
    for i in idx:
        images.append(images_prev[i])
        filenames.append(images_names[i])

    if as_array:
        images = np.stack(images, axis=0)

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

def save_images(path, images, filenames):
    """ saves images to a specific path:

        - path is the path to the directory in which the images
            should be saved
        - images is a list/numpy.ndarray of images to save
        - filenames is a list of filenames of the images to save

        Returns: True on success and False on failure
    """
    #try:
    print("hola1")
    os.chdir(path)
    print("hola2")
    for i, name in enumerate(filenames):
        print("hola3")
        cv2.imwrite(name, images[i])
    return True
    """
    except:
        return False
    """

def generate_triplets(images, filenames, triplet_names):
    """ generates triplets:

        - images is a numpy.ndarray of shape (n, h, w, 3) containing
            the various images in the dataset
        - filenames is a list of length n containing the corresponding
            filenames for images
        - triplet_names is a list of lists where each sublist contains
            the filenames of an anchor, positive, and negative image,
            respectively

    Returns: a list [A, P, N]

        - A is a numpy.ndarray of shape (m, h, w, 3) containing the
            anchor images for all m triplets
        - P is a numpy.ndarray of shape (m, h, w, 3) containing the
            positive images for all m triplets
        - N is a numpy.ndarray of shape (m, h, w, 3) containing the
            negative images for all m triplet
    """
    a_names = [triplet[0] for triplet in triplet_names]
    p_names = [triplet[1] for triplet in triplet_names]
    n_names = [triplet[2] for triplet in triplet_names]

    ind_A = list(filter(lambda i: filenames[i] in a_names, range(len(filenames))))
    ind_P = list(filter(lambda i: filenames[i] in p_names, range(len(filenames))))
    ind_N = list(filter(lambda i: filenames[i] in n_names, range(len(filenames))))

    return [images[ind_A], images[ind_P], images[ind_N]]
