#!/usr/bin/env python
# -*- coding: utf-8 -*-

from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

# load the example image
image = cv2.imread("frame1323.jpg")
# image = cv2.imread("example.jpg");
rotated = imutils.rotate(image, 270)
# cv2.imshow("Rotated", rotated)
# cv2.waitKey(0)

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(rotated, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

# show image after edged
# cv2.imshow('image edged', edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 在边缘图中，从高到低给轮廓排序
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 判断轮廓是否有四个角
    if len(approx) == 4:
        displayCnt = approx
        break

# apply a perspective transform
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
output = four_point_transform(image, displayCnt.reshape(4, 2))
output_sized = output[37:102, 52:225]




# 转换成黑白图像
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# rows, cols = thresh.shape
# 旋转图像，获取数字区域，反转图片
# M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1)
# dst = cv2.warpAffine(thresh, thresh, (cols, rows))
thresh = thresh[37:102, 55:225]
thresh_black = cv2.bitwise_not(thresh)

# to connect parts in one digit
# kernel = np.ones((5,5),np.uint8)
# thresh = cv2.dilate(thresh,kernel,iterations = 1)

# find contours in the thresholded image, then initialize the digit contours lists
contour, hierarchy = cv2.findContours(thresh_black.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


cnts = cv2.findContours(thresh_black.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []

# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    # if the contour is sufficiently large, it must be a digit

    if h >= 18:
        digitCnts.append(c)

    # if w >= 15 and (h >= 30 and h <= 40):
    #     digitCnts.append(c)

# sort the contours from left-to-right, then initialize the actual digits themselves
digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
digits = []

# loop over each of the digits
for c in digitCnts:
    counter = 0
    # extract the digit ROI, because of 1, need to hardcode define roi
    (x, y, w, h) = cv2.boundingRect(c)
    current_digit_index = counter
    roi = thresh_black[y:y + h, 40 * (current_digit_index):40 * (current_digit_index+1)]
    #roi = thresh[y:y + h, x:x + w]

    # cv2.imshow('image edged', roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # compute the width and height of each of the 7 segments
    # we are going to examine
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)

    # define the set of 7 segments
    segments = [
        ((0, 0), (roiW, dH)),  # top
        ((0, 0), (dW, h // 2)),  # top-left
        ((roiW - dW, 0), (roiW, h // 2)),  # top-right
        ((0, (h // 2) - dHC), (roiW, (h // 2) + dHC)),  # center
        ((0, h // 2), (dW, h)),  # bottom-left
        ((roiW - dW, h // 2), (roiW, h)),  # bottom-right
        ((0, h - dH), (roiW, h))  # bottom
    ]
    on = [0] * len(segments)

    # loop over the segments
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)

        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) > 0.5:
            on[i] = 1

    # lookup the digit and draw it on the image
    digit = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit)
    # cv2.rectangle(output_sized, (x, y), (x + roiW, y + h), (0, 255, 0), 1)
    # cv2.putText(output_sized, str(digit), (x - 10, y - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    counter += 1
    
# display the digits
print(u"{}{}{}.{} \u00b0C".format(*digits))
cv2.imshow("Input", image)
cv2.imshow("Output", output_sized)
cv2.waitKey(0)