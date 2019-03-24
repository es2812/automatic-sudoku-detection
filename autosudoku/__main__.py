import cv2
from os.path import split, join, realpath
import argparse
from math import sqrt
from pipe import Pipe
import numpy as np
from show import show
from PIL import ImageFont, Image, ImageDraw

def main(path):
    """ Takes the path to an image with a sudoku puzzle and uses the class Pipe to obtain the matrix that represents said sudoku, outputting it to console and showing the original image with the matrix overlaid.

    Args:
        path: string
    """
    pipe = Pipe()
    image = cv2.imread(path)
    try:
        sudoku = pipe.extractNumbers(pipe.warpSudoku(image))
    except ValueError as err:
        print(err)
        return
    image = pipe.getImage()

    #to demostrate the functionality we draw the result matrix on top of the original picture
    #we turn the matrix into a string formatted to our taste
    sudoku_str = ""
    square_size = sqrt(len(sudoku[0]))

    for row_ix,i in enumerate(sudoku):
        if(row_ix != 0 and row_ix % square_size == 0):
            sudoku_str += "\n"

        for column_ix,j in enumerate(i):
            
            if(column_ix != 0 and column_ix % square_size == 0):
                sudoku_str += "    "
            if(j == -1):
                sudoku_str += "- "
            else:
                sudoku_str += str(j) + " "
            

        sudoku_str += "\n"
    print(sudoku_str)
    # opencv works with BGR, Pillow with RGB
    img_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #we pass the image to Pillow
    pil_image = Image.fromarray(img_converted)
    #we use a new font
    font_path = join(split(realpath(__file__))[0],"UbuntuMono-B.ttf")
    font = ImageFont.truetype(font_path, int(0.03*image.shape[1]))
    #we draw the text
    draw = ImageDraw.Draw(pil_image)  
    #we'll draw it in the middle of the picture
    H = image.shape[0]
    W = image.shape[1]
    w,h = draw.textsize(sudoku_str, font=font)
    x = int((W-w)/2)
    y = int((H-h)/2)
    bw = 2
    #with outline
    draw.text((x+bw,y), sudoku_str, font=font, fill=(0,0,0))
    draw.text((x-bw,y), sudoku_str, font=font, fill=(0,0,0))
    draw.text((x,y-bw), sudoku_str, font=font, fill=(0,0,0))
    draw.text((x,y+bw), sudoku_str, font=font, fill=(0,0,0))
    draw.text((x,y), sudoku_str, font=font, fill=(255,0,0))
    #get the image back
    img_drawn = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    show('result',img_drawn)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image",help="Path to the image")

    args = parser.parse_args()
    path = args.image

    main(path)
