# This class implements a rectangle detector.
# It takes a series of contours and returns the biggest rectangle
# found in the image as a series of vertices
import cv2
import numpy as np

class RectangleDetector:
    def __init__(self):
        self.full_approximation = []
        self.rectangles = []

    def getPolyApprox(self):
         return self.full_approximation

    def getRectangles(self):
        return self.rectangles

    def __approximate__(self,cont):
        self.full_approximation = []
        self.rectangles = []
        dist_allowed = 0.04
        while(len(self.rectangles)==0):
            for c in cont:
                peri = cv2.arcLength(c,True)
                approx = cv2.approxPolyDP(c,dist_allowed*peri,True)
                self.full_approximation.append(approx)
                if(len(approx)==4):
                    self.rectangles.append(approx)
            dist_allowed += 0.01 #we diminish the quality of the approximation until we find a rectangle
            if(dist_allowed == 1):
                break #if we have reached a 100% quality we stop, it won't get any better
        return self.rectangles

    def __getBiggestShape__(self,shapes):
        areas = [cv2.contourArea(shape) for shape in shapes]
        biggest_figure_index = [i for i,a in enumerate(areas) if a==max(areas)][0]
        return shapes[biggest_figure_index]
    
    
    def getBiggestRectangle(self,contours):
        return self.__getBiggestShape__(self.__approximate__(contours))