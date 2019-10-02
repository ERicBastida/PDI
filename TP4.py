# GUIA PRACTICA N4 - COLOR


import cv2
import numpy as np
from matplotlib import pyplot as plt

img_o = cv2.imread('img/rosas.jpg',1)
# img = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
# b,g,r = cv2.split(img)
# print img.shape

plt.figure(1)
plt.subplot(111)
plt.imshow(img_o)
plt.title('Imagen original')

plt.figure(2)
hsv = cv2.cvtColor(img_o, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
plt.subplot(131)
plt.imshow(h,cmap='gray')
plt.title('Hue')

plt.subplot(132)
plt.imshow(s,cmap='gray')
plt.title('Saturation')

plt.subplot(133)
plt.imshow(v,cmap='gray')
plt.title('Value')

plt.show()


#
# # define range of blue color in HSV
# lower_red = np.array([173,0,30])
# upper_red = np.array([176,50,100])
# #
# # # Threshold the HSV image to get only blue colors
# mask = cv2.inRange(img, lower_red, upper_red)
# plt.subplot(122)
# plt.imshow(mask,cmap='gray')
# plt.title('MASK')
# #
# # # Bitwise-AND mask and original image
# # res = cv2.bitwise_and(frame,frame, mask= mask)
# #
# # cv2.imshow('frame',frame)
# # cv2.imshow('mask',mask)
# # cv2.imshow('res',res)
# # k = cv2.waitKey(5) & 0xFF
# # if k == 27:
# #     break
#
# plt.show()
#


