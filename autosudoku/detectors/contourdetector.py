#This class implements a ContourDetector
#it takes a grayscaled image, and calculates the contours by
#thresholding the image with the Adaptive Gaussian method

import cv2
import numpy as np
from show import show

class ContourDetector:
    def __init__(self):
        self.threshold = []

    def __threshold__(self,img,method):
        if(method=='otsu'):
            _,self.threshold = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif(method=='gaussian'):
            self.threshold = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        self.threshold = cv2.bitwise_not(self.threshold)
        return self.threshold

    def getThresholded(self):
        return self.threshold

    def getContours(self,img,include_internal=True,thresholding='otsu'):
        if(include_internal):
            return cv2.findContours(self.__threshold__(img,thresholding), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            return cv2.findContours(self.__threshold__(img,thresholding), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)