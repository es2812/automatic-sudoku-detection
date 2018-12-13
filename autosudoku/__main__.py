import cv2
import argparse
from pipe import Pipe

def show(name,image):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.imshow(name,image)

parser = argparse.ArgumentParser()
parser.add_argument("image",help="Path to the image")

args = parser.parse_args()

filename = args.image
image = cv2.imread(filename,1)
show('window',image)

pipe = Pipe()
puzzle = pipe.warpSudoku(image)

show('Puzzle',puzzle)

cv2.waitKey(0)
cv2.destroyAllWindows()
exit()