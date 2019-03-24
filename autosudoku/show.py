import cv2

def show(name,image):
    """Creates a resizable window and shows an image with a given window name.

    Args:
        name: string
        image: numpy.ndarray
    """
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.imshow(name,image)