#This class implements a number classifier
#it uses a previously trained TensorFlow model (.pb file)
#and takes an array of regions of interest as input, which it then transforms into
#valid inputs to the model and returns the output classes

import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import os
import cv2
import numpy as np

def __preprocess(img):
    """Processes a given ROI for input into the network.

    It resizes the ROI to 20x20 and adds a 4px border to make a centered 28x28px image, which then normalizes.

    Args:
        img: np.ndarray

    Returns:
        prepared_img: np.ndarray
    """
    h,w = img.shape[:2]
    img = cv2.resize(img, (20, 20), interpolation = cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, 4 , 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    # we transform from [0,255] to [0.0,1.0] range
    return img / 255

def classifynumbers(images, model_relative_path="models/char74k"):
    """Implements a number classifier.

    It uses a previously trained TensorFlow model (saved_model.pb file) and takes an array of regions of interest as input, which it then transforms into valid inputs to the model and returns the output classes.

    Args:
        images: np.ndarray
        model_relative_path - location of the TenforFlow model (must be a saved_model.pb file): string (default: 'models/char74k')

    Returns:
        output: np.ndarray
    """
    filename = os.path.join(os.path.split(os.path.realpath(__file__))[0],model_relative_path)
    #we start a tensorflow session
    with tf.Session(graph=tf.Graph()) as sess:
        #we import the graph found in the path
        tf.saved_model.loader.load(sess, ['serve'], filename)
        #we find the input and output tensors
        x = sess.graph.get_tensor_by_name('Placeholder:0')
        y = sess.graph.get_tensor_by_name('ArgMax:0')
        images = np.array(list(map(__preprocess, images)))
        y_out = sess.run(y,feed_dict={x:images})
        sess.close()
    return y_out