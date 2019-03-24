import cv2
import numpy as np

class RectangleDetector:
    """This class implements a rectangle detector.

    It takes a series of contours and returns the biggest rectangle found in the image as a series of vertices.
    """
    def __init__(self):
        self.full_approximation = []
        self.rectangles = []

    def getRectangles(self):
        """Returns the closed 4th degree polynomials approximated to the contours.

        Returns:
            rectangles: numpy.ndarray
        """
        return self.rectangles

    def __rectangles__(self,cont):
        """Returns all closed 4th degree polynomials found approximating the contours.

        The function degrades the approximation's accuracy until it founds at least one 4th degree polynomial.

        Args:
            cont: numpy.ndarray

        Returns:
            rectangles: numpy.ndarray
        """
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
        """Returns the biggest shape (by area)

        Args:
            shapes: numpy.ndarray

        Returns:
            biggest_shape: numpy.ndarray
        """
        areas = [cv2.contourArea(shape) for shape in shapes]
        biggest_figure_index = [i for i,a in enumerate(areas) if a==max(areas)][0]
        return shapes[biggest_figure_index]
    
    def getShapes(self,contours):
        """Calculates all closed shapes approximated to the contours by an epsilon of 0.04

        Args:
            contours: numpy.ndarray

        Returns:
            shapes: numpy.ndarray 
        """
        result = list()
        for c in contours:
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,0.04*peri,True)
            result.append(approx)
        return result

    def getBiggestRectangle(self,contours):
        """Returns the biggest rectangle found by approximating the contours to closed 4th degree polynomials.

        Args:
            contours: numpy.ndarray

        Returns:
            biggest_rectangle: numpy.ndarray
        """
        return self.__getBiggestShape__(self.__rectangles__(contours))