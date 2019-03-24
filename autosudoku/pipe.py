import cv2
import numpy as np
from detectors.rectangledetector import RectangleDetector
from detectors.contourdetector import ContourDetector
from classifiers.numberclassifier import classifynumbers
from math import sqrt


class Pipe:
    """
    Class that implements a Computer Vision Pipeline, which detects and returns a sudoku in matrix form from a picture.

    It involves two steps, signified by the public functions with the same name:
        - warpSudoku: takes the original image and generates a new one that contains only the found sudoku puzzle.
        - extractNumbers: takes a picture containing ONLY a sudoku puzzle and returns the sudoku in matrix form, with the empty spaces replaced with -1.
    """
    def __init__(self):
        self.image = []
        self.contours = []
        self.rect = []
        self.puzzle = []

        self.contourer = ContourDetector()
        self.rectangler = RectangleDetector()

    def __orderPoints__(self,points):
        """Orders a set of four points clockwise starting from the upper left point

        Args:
            points: numpy.ndarray
    
        Returns:
            sorted_points: numpy.ndarray
        """
        #ordering by x or y coordinates gives the left/right most and top/bottom most points respectively
        xSorted = points[np.argsort(points[:,0]),:]
        lefts = xSorted[:2,:]
        rights = xSorted[2:,:]

        top_left, bottom_left = lefts[np.argsort(lefts[:,1]),:]
        top_right, bottom_right = rights[np.argsort(rights[:,1]),:]
    
        return np.array([top_left, top_right, bottom_right, bottom_left],dtype=np.float32)

    def getImage(self):
        """Returns original image used by this instance of Pipe
        
        Returns:
            image: numpy.ndarray
        """
        return self.image

    def getThresholded(self):
        """Returns image after applying thresholding
        
        Returns:
            thresholded: numpy.ndarray
        """
        return self.contourer.getThresholded()

    def getContours(self):
        """Returns contours found in the image
        
        Returns:
            contours: numpy.ndarray
        """
        return self.contours

    def getPolyApprox(self):
        """Returns the closed polinomial approximations to the contours found

        Returns:
            polynomials: numpy.ndarray
        """
        return self.rectangler.getPolyApprox()

    def getRectangles(self):
        """Returns all closed 4th degree polynomials found after approximating the contours

        Returns:
            rectangles: numpy.ndarray
        """
        return self.rectangler.getRectangles()

    def getBiggestRectangle(self):
        """Returns the largest (by area) closed 4th degree polynomial found after approximating the contours

        Returns:
            rectangles: numpy.ndarray
        """
        return self.rect

    def warpSudoku(self,image):
        """Trims and warps (if necessary) an image containing a sudoku puzzle to only contain said puzzle.
        
        Args:
            image: numpy.ndarray

        Returns:
            sudoku: numpy.ndarray
        """
        self.image = image
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
        self.puzzle =  cv2.warpPerspective(copy, M, (new_width, new_height))
        return self.puzzle

    def __getROI__(self,c):
        """Returns a ROI taken from the threshold bounding a given contour

        Args:
            c - contour: numpy.ndarray

        Returns:
            roi: numpy.ndarray
        """
        #we make a straight bounding rectangle around the number
        x,y,w,h = cv2.boundingRect(c) #coordinates of top-left, width and height
        thresh = self.contourer.getThresholded()
        return cv2.bitwise_not(thresh[y:y+h,x:x+w])

    def __getNumbers__(self,c):
        """Returns the numbers in the contours as classified by the model defined in classifiers.numberclassifiers

        Args:
            c - contours: numpy.ndarray

        Returns:
            number: int
        """
        rois = np.array(list(map(self.__getROI__, c)))
        return classifynumbers(rois)

    def extractNumbers(self,img):
        """Takes an image containing ONLY a sudoku puzzle and extracts the matrix that represents it, a two-dimensional numpy array with integers or -1 in the case of empty spaces.

        Args:
            img: numpy.ndarray

        Returns:
            sudoku: numpy.ndarray

        Raises:
            ValueError: No contours with the characteristics that a sudoku should follow were found 
        """

        #takes the warped image and extracts all numbers in order left-right/top-bottom

        #(1) we grayscale the image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #(2) we apply gaussian blur to smooth the edges and reduce noise
        gaussian = cv2.GaussianBlur(gray,(5,5),0)
        
        #(3) we detect contours
        _, contours, hierarchy = self.contourer.getContours(gaussian,include_internal=True,thresholding='otsu')
        
        #(4) we separate the contours according to hierarchy

        #hierarchy legend: [Next, Previous, First_Child, Parent]
        empty = list()
        contain_number = list()
        number_contained = list() #will help us find the number later
        numbers = list()

        for index,(_,_,child,parent) in enumerate(hierarchy[0]):
            if(parent==0):
                if(child!=-1):
                    contain_number.append(contours[index])
                    number_contained.append(child)
                else:
                    empty.append(contours[index])

        #(5) we approximate empty and contain_number with polydp
        ap_empty = self.rectangler.getShapes(empty)
        ap_contain = self.rectangler.getShapes(contain_number)

        #(6) we get the minimum X and Y coordinates for each contour
        mins = list()
        all_squares = ap_empty+ap_contain
        if(len(all_squares)>0):
            for verts in all_squares:
                xMin = np.amin(verts[:,0,0])
                yMin = np.amin(verts[:,0,1])
                mins.append([xMin,yMin])
        
            mins = np.array(mins)
            sorted_indices = np.argsort(mins[:,1])
            
            #we assume a square matrix and that all squares have been found
            matrix_size = int(sqrt(len(all_squares)))

            sudoku = np.zeros((matrix_size, matrix_size),dtype=np.int)
            number_contours = {'i':[],'j':[],'c':[]}
            for i in range(0,matrix_size):
                indices_row = sorted_indices[i*matrix_size:(i+1)*matrix_size]
                #we xSort the vertices in the row
                indices_row = indices_row[np.argsort(mins[indices_row,0])]
                #we find out wether the index is in ap_empty
                #and we give that square -1
                for j,r in enumerate(indices_row):
                    if r < len(ap_empty):
                        sudoku[i,j] = -1
                    else:
                        #we have the number_contained array which contains the index of the contour that represents the number that is contained inside each non-empty square
                        c = contours[number_contained[r-len(ap_empty)]]
                        number_contours['i'].append(i)
                        number_contours['j'].append(j)
                        number_contours['c'].append(c)

            
            numbers = self.__getNumbers__(number_contours['c'])
            #now we can place each number in its place
            for idx,row in enumerate(number_contours['i']):
                i = row
                j = number_contours['j'][idx]
                n = numbers[idx]
                sudoku[i,j] = n

            return sudoku
        else:
            raise ValueError("No valid contours found")
