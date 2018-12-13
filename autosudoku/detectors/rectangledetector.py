# This class implements a rectangle detector.
# It takes a series of contours and returns the biggest rectangle
# found in the image as a series of vertices
import cv2
import numpy as np

class RectangleDetector:
    def __init__(self):
        pass

    def __approximate__(self,cont):
        rectangles = []
        for c in cont:
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,0.04*peri,True)
            if(len(approx)==4):
                rectangles.append(approx)
        return rectangles

    def __getBiggestShape__(self,shapes):
        areas = [cv2.contourArea(shape) for shape in shapes]
        biggest_figure_index = [i for i,a in enumerate(areas) if a==max(areas)][0]
        return shapes[biggest_figure_index]
    
    
    def getBiggestRectangle(self,contours):
        return self.__getBiggestShape__(self.__approximate__(contours))