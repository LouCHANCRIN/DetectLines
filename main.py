import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = np.shape(image)[0]
    y2 = int(y1 * (2/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    ret = np.array([x1, y1, x2, y2])
    return (ret)

def average_slope(image, lines):
    left_fit = []
    right_fit = []
    count_left = 0
    count_right = 0
    for x1, y1, x2, y2 in lines:
        #np.polyfit give the slope and the y intercept of the line
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        if (slope < 0):
            left_fit.append((slope, intercept))
            count_left += 1
        else:
            right_fit.append((slope, intercept))
            count_right += 1

    if (count_right == 0 or count_left == 0):
        return (lines)
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return (np.array([left_line, right_line]))

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
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return (line_image)

def main_image():
    color_image = cv2.imread("test_image.jpg")
    image = canny(color_image)

    #using plt make it easy to find the coordinate of the region of intersest
    #plt.imshow(image)
    #plt.show()
    roi = region_of_interest(image)

    #Hough transform find the line that best describe our points
    #2 mean 2pixel precision
    #using np.pi to have a precision of 1 degree in radian
    #100 is the threshold of vote per bin
    #minLineLength=X means that we dont accept line smaller than X pixels
    #maxLineGap=X means that if 2 line have a gap <=X pixels we can connect them in a single line
    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]),
            minLineLength=40, maxLineGap=200)

    lines = np.reshape(lines, (np.shape(lines)[0], 4))
    averaged_line = average_slope(color_image, lines)
    line_image = create_line_image(color_image, averaged_line)
    image_with_line = cv2.addWeighted(color_image, 0.8, line_image, 1, 1)
    cv2.imshow("Image", image_with_line)
    cv2.waitKey(0)

def main_video():
    vid = cv2.VideoCapture("test2.mp4")
    print("press q to quit")
    while (vid.isOpened()):
        ret, frame = vid.read()
        if (ret == False):
            break
        image = canny(frame)
        roi = region_of_interest(image)
        lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]),
                minLineLength=40, maxLineGap=10)
        lines = np.reshape(lines, (np.shape(lines)[0], 4))
        averaged_line = average_slope(frame, lines)
        line_image = create_line_image(frame, averaged_line)
        image_with_line = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("Video", image_with_line)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_video()
    #main_image()
