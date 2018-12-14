import cv2
import numpy as np
from detectors.rectangledetector import RectangleDetector
from detectors.contourdetector import ContourDetector

class Pipe:
    def __init__(self):
        pass

    def __orderPoints__(self,points):
        #ordering by x or y coordinates gives the left/right most and top/bottom most points respectively
        xSorted = points[np.argsort(points[:,0]),:]
        lefts = xSorted[:2,:]
        rights = xSorted[2:,:]

        top_left, bottom_left = lefts[np.argsort(lefts[:,1]),:]
        top_right, bottom_right = rights[np.argsort(rights[:,1]),:]
    
        return np.array([top_left, top_right, bottom_right, bottom_left],dtype=np.float32)

    def warpSudoku(self,filename):
        #(0) we read the image
        image = cv2.imread(filename)

        #(1) we grayscale the image
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        #(2) we apply gaussian blur to smooth the edges and reduce noise
        gaussian = cv2.GaussianBlur(gray,(5,5),0)

        contourer = ContourDetector()
        rectangler = RectangleDetector()

        #(3) we detect contours
        contours = contourer.getContours(gaussian,include_internal=False) #only external contours
        
        #(4) we approximate a polydp and keep the rectangle
        rect = rectangler.getBiggestRectangle(contours)
    
        # (5) The approximation obtained is the vertices of the sudoku puzzle
        # we can use them to do a affine transformation
        new_size = int(cv2.arcLength(rect,True)/4) #we use one fourth of the figure to roughly obtain the side's length (we assume it's a square)
        rect = rect[:,0,:] #to 2-dimensional array
        
        # We want to order the points clockwise: top-left, top-right, bottom-right, bottom-left
        src = self.__orderPoints__(rect)
        dst = np.float32([[0,0],[new_size,0],[new_size,new_size],[0,new_size]]) #we want the sudoku to occupy the whole image
        M = cv2.getPerspectiveTransform(src,dst)
        return cv2.warpPerspective(image, M, (new_size, new_size))