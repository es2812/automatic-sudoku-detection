import cv2
import numpy as np
from detectors.rectangledetector import RectangleDetector
from detectors.contourdetector import ContourDetector

class Pipe:
    def __init__(self):
        self.image = []
        self.contours = []
        self.rect = []

        self.contourer = ContourDetector()
        self.rectangler = RectangleDetector()

    def __orderPoints__(self,points):
        #ordering by x or y coordinates gives the left/right most and top/bottom most points respectively
        xSorted = points[np.argsort(points[:,0]),:]
        lefts = xSorted[:2,:]
        rights = xSorted[2:,:]

        top_left, bottom_left = lefts[np.argsort(lefts[:,1]),:]
        top_right, bottom_right = rights[np.argsort(rights[:,1]),:]
    
        return np.array([top_left, top_right, bottom_right, bottom_left],dtype=np.float32)

    def getImage(self):
        return self.image

    def getThresholded(self):
        return self.contourer.getThresholded()

    def getContours(self):
        return self.contours

    def getPolyApprox(self):
        return self.rectangler.getPolyApprox()

    def getRectangles(self):
        return self.rectangler.getRectangles()

    def getBiggestRectangle(self):
        return self.rect

    def warpSudoku(self,filename):
        #(0) we read the image
        self.image = cv2.imread(filename)

        #(1) we grayscale the image
        gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)

        #(2) we apply gaussian blur to smooth the edges and reduce noise
        gaussian = cv2.GaussianBlur(gray,(5,5),0)
        
        #(3) we detect contours
        # we have two methods implemented, otsu and gaussian
        _, self.contours,_ = self.contourer.getContours(gaussian,include_internal=False,thresholding='otsu')

        #(4) we approximate a polydp and keep the rectangle
        self.rect = self.rectangler.getBiggestRectangle(self.contours)
    
        # (5) The approximation obtained is the vertices of the sudoku puzzle
        # we can use them to do a perspective transformation
        self.rect = self.rect[:,0,:] #to 2-dimensional array
        
        # We want to order the points clockwise: top-left, top-right, bottom-right, bottom-left
        src = self.__orderPoints__(self.rect)
        new_width = max(abs(src[1,0]-src[0,0]),abs(src[2,0]-src[3,0]))
        new_height = max(abs(src[0,1]-src[3,1]),abs(src[1,1]-src[2,1]))
        dst = np.float32([[0,0],[new_width,0],[new_width,new_height],[0,new_height]]) #we want the sudoku to occupy the whole image
        M = cv2.getPerspectiveTransform(src,dst)
        copy = self.image.copy()
        return cv2.warpPerspective(copy, M, (new_width, new_height))