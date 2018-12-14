# This scripts takes a directory containing pictures of sudokus
# and applies the Pipe class to each, which attempts to detect the puzzle.
# 
# It then shows the result on screen and allows the user to press
# the keys 'g' (for good) and 'b' (for bad) on their keyboard or 'q' (for quit) to abort the process.
#
# When it  has gone through all the images it moves the images classified as bad to a folder
# named 'problematic' that MUST exist inside the folder with which this script is called.
import cv2
import argparse
from pipe import Pipe
from os import walk, rename
from os.path import join, split
from shutil import move

def show(name,image):
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.imshow(name,image)

parser = argparse.ArgumentParser()
parser.add_argument("dir",help="Path to the directory containing images")

args = parser.parse_args()
path = args.dir

f = []
for (dirpath, dirnames, filenames) in walk(path):
    for file in filenames:
        full_path = join(dirpath,file)
        f.append(full_path)
    
badly_approximated = list()
for filename in f:
    pipe = Pipe()
    puzzle = pipe.warpSudoku(filename)

    show(filename,puzzle)

    k = cv2.waitKey(0)
    if(k==ord('q')):
        #quitting process
        break
        print(k)
        print(ord('g'))
    
    while(1):
        if k==ord('g'):
            #the algorithm has correctly warped the sudoku
            print(filename + " is a good sudoku")
            break

        elif k==ord('b'):
            #the algorithm hasn't correctly warped the sudoku. we save the path
            print(filename + " is a bad sudoku")
            badly_approximated.append(filename)
            break
        else:
            k = cv2.waitKey(0)

    cv2.destroyAllWindows()
#we take all of the pics in the badly_approximated list and move them over to the "problematic" folder
#inside the directory with which this script was called (MUST EXIST)
for b in badly_approximated:
    head,file = split(b)
    new_path = join(path,"problematic",file)
    move(b, new_path)
exit()