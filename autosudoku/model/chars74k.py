# This file prepares the dataset for usage in the minst classifier by tensorflow. Steps are as follows:
# 1) Assign label, recognizable by the folder they're in
# 2) Read image to numpy matrix through opencv
# 3) Resize image to 28x28
# 4) Convert numpy matrix type to float32
# 5) Normalize data to [0.0,1.0] range
# 6) Create indexes for shuffle-split of dataset in training and test sets
# 7) train and test functions return those splits in tf.data.Dataset format

import cv2
import os
import numpy as np
import tensorflow as tf
class Chars74K:
    """Prepares the dataset Chars74K for usage in the mnist classifier by tensorflow.

    Steps are as follows:
        1) Assign label, recognizable by the folder they're in
        2) Read image to numpy matrix through opencv
        3) Resize image to 28x28
        4) Convert numpy matrix type to float32
        5) Normalize data to [0.0,1.0] range
        6) Create indexes for shuffle-split of dataset in training and test sets
    """
    images = list()
    labels = list()

    def __init__(self):
        dirs = ['EnglishFnt/Sample002','EnglishFnt/Sample003','EnglishFnt/Sample004','EnglishFnt/Sample005','EnglishFnt/Sample006','EnglishFnt/Sample007','EnglishFnt/Sample008','EnglishFnt/Sample009','EnglishFnt/Sample010']

        path = os.path.split(os.path.realpath(__file__))[0]
        posslabels = [1,2,3,4,5,6,7,8,9]


        def decode_image(filename):
            print("Decoding %s" % filename)
            img = cv2.imread(filename,0)
            img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_LINEAR)
            img = img.astype('float32')
            return img/255.0

        for i,dir in enumerate(dirs):
            label = posslabels[i]

            for file in os.listdir(dir):
                self.images.append(decode_image(os.path.join(path,dir,file)))
                self.labels.append(label)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        #shuffle split of indexes
        idx = np.arange(0,len(self.images))
        np.random.shuffle(idx)
        split = int((2/3)*len(self.images))
        self.train_idx = idx[0:split]
        self.test_idx = idx[split:]

    def train(self):
        """Returns dataset for training.

        Returns:
            ds_train: tensorflow.data.Dataset
        """
        x = self.images[self.train_idx]
        y = self.labels[self.train_idx]

        return tf.data.Dataset.from_tensor_slices(({'image':x},y))

    def test(self):
        """Returns dataset for validation.

        Returns:
            ds_test: tensorflow.data.Dataset
        """
        x = self.images[self.test_idx]
        y = self.labels[self.test_idx]

        return tf.data.Dataset.from_tensor_slices(({'image':x},y))
    