import cv2
import numpy as np

class ContourDetector:
    """This class implements a ContourDetector. 
    
    It takes a grayscaled image, and calculates the contours by thresholding the image with the Adaptive Gaussian or Otsu methods.
    """
    def __init__(self):
        self.threshold = []

    def __threshold__(self,img,method):
        """Obtains the thresholded image.

        Args:
            img: numpy.ndarray
            method: string {'otsu','gaussian'}

        Returns:
            thresholded_img: numpy.ndarray
        """
        if(method=='otsu'):
            _,self.threshold = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif(method=='gaussian'):
            self.threshold = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        self.threshold = cv2.bitwise_not(self.threshold)
        return self.threshold

    def getThresholded(self):
        """Returns the thresholded image previously calculated

        Returns:
            thresholded_img: numpy.ndarray
        """
        return self.threshold

    def getContours(self,img,include_internal=True,thresholding='otsu'):
        """Obtains the contours of the image.

        Args:
            img: numpy.ndarray
            include_internal - whether or not to include internal contours. If False returns hierarchy in Tree form. bool (default=True)
            method - method to use for thresholding. string {'otsu','gaussian'} (default='otsu')

        Returns:
            img: numpy.ndarray
            contours: numpy.ndarray
            hierarchy: numpy.ndarray
        """


        if(include_internal):
            return cv2.findContours(self.__threshold__(img,thresholding), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            return cv2.findContours(self.__threshold__(img,thresholding), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)