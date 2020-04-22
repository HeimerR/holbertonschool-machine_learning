#!/usr/bin/env python3
""" class face aling """


class FaceAlign:
    """ Initialize Face Align """
    def __init__(self, shape_predictor_path):
        """ init for class
            shape_predictor_path is the path to the dlib shape predictor model
            detector - contains dlib‘s default face detector
            shape_predictor - contains the dlib.shape_predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(args[shape_predictor_path])


    def detect(self, image):
        """ detects a face in an image:
            image is a numpy.ndarray of rank 3 containing an image from
            which to detect a face

            Returns: a dlib.rectangle containing the boundary box for the face
            in the image, or None on failure
            If multiple faces are detected, it returns the dlib.rectangle
            with the largest area
            If no faces are detected, it returns a dlib.rectangle that is the
            same as the image
        """
        faces = self.detector(image, 1)
	x = faces.left()
	y = faces.top()
	w = faces.right() - x
	h = faces.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)
