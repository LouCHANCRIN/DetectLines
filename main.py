import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(color_image):
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    #Using gaussian blur to reduce the noise in the image
    image = cv2.GaussianBlur(image, (5, 5), 0)

    #cv2.Canny is computing the derivative between adjacent pixels to get
    #the change in intensity to find the edges.
    image = cv2.Canny(image, 50, 150)
    return (image)

def region_of_interest(image):
    #creating a black image and adding a white triangle on the region of
    #interest to make a bitwise operation to keep only the ROI edge
    height = np.shape(image)[0]
    roi = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return (masked_image)

def create_line_image(image, lines):
    line_image = np.zeros_like(image)
    if (lines is not None):
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return (line_image)


def main():
    color_image = cv2.imread("test_image.jpg")
    image = canny(color_image)

    #using plt make it easy to find the coordinate of the region of intersest
    #plt.imshow(image)
    #plt.show()
    roi = region_of_interest(image)

    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]),
            minLineLength=40, maxLineGap=5)#using np.pi to have radians
    line2 = create_line_image(color_image, lines)
    image_with_line = cv2.addWeighted(color_image, 0.8, line2, 1, 1)

    cv2.imshow("image", image_with_line)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
