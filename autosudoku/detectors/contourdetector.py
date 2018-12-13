#This class implements a ContourDetector
#it takes a grayscaled image, and calculates the contours by
#thresholding the image with the OTSU method

import cv2
import numpy as np

class ContourDetector:
    def __init__(self):
        pass

    def __threshold__(self,img):
        _,threshold = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return cv2.bitwise_not(threshold)

    def getContours(self,img,include_internal=True):
        if(include_internal):
            return cv2.findContours(self.__threshold__(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        else:
            return cv2.findContours(self.__threshold__(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]