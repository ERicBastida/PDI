# GUIA PRACTICA N4 - COLOR


import cv2
import numpy as np
from matplotlib import pyplot as plt

class TP4:
    __basePath = "img/TP4 - Color/"

    def ejercicio1(self):
        img = cv2.imread(self.__basePath+ "patron.tif")
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgHSV = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2HSV)
        R, G, B = cv2.split(imgRGB)
        H,S,V = cv2.split(imgHSV)

        plt.figure("Figura Original")
        plt.imshow(imgRGB)

        plt.figure("Componentes RGB")                       
        plt.subplot(231),plt.imshow(R,cmap='gray',vmin=0,vmax=255),plt.title("Red")
        plt.subplot(232),plt.imshow(G,cmap='gray',vmin=0,vmax=255),plt.title("Green")
        plt.subplot(233),plt.imshow(B,cmap='gray',vmin=0,vmax=255),plt.title("Blue")
        
        plt.subplot(234),plt.imshow(H,cmap='gray',vmin=0,vmax=255),plt.title("Hue")
        plt.subplot(235),plt.imshow(S,cmap='gray',vmin=0,vmax=255),plt.title("Saturation")
        plt.subplot(236),plt.imshow(V,cmap='gray',vmin=0,vmax=255),plt.title("Value")


        plt.figure("PROCESADO")

        # G = np.uint8(np.zeros(G.shape))
        R =  255*np.uint8(np.ones(V.shape)) - R
        G =  255*np.uint8(np.ones(V.shape)) - G
        B =  255*np.uint8(np.ones(V.shape)) - B
        imgNEW_RGB_withoutGreen = cv2.merge((R,G,B))

        # V = V + 100*np.uint8(np.ones(V.shape))
        H = 255*np.uint8(np.ones(H.shape)) - H
        imgNEW_HSI = cv2.merge((H,S,V))
        imgNEW_RGB_fixedIntensity = cv2.cvtColor(imgNEW_HSI,cv2.COLOR_HSV2RGB)

        plt.subplot(121)
        plt.imshow(imgNEW_RGB_withoutGreen)
        plt.subplot(122)
        plt.imshow(imgNEW_RGB_fixedIntensity)
        plt.show()
    



tp4 = TP4()

tp4.ejercicio1()
        



# img_o = cv2.imread('img/rosas.jpg',1)
# # img = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
# # b,g,r = cv2.split(img)
# # print img.shape

# plt.figure(1)
# plt.subplot(111)
# plt.imshow(img_o)
# plt.title('Imagen original')

# plt.figure(2)
# hsv = cv2.cvtColor(img_o, cv2.COLOR_BGR2HSV)
# h,s,v = cv2.split(hsv)
# plt.subplot(131)
# plt.imshow(h,cmap='gray')
# plt.title('Hue')

# plt.subplot(132)
# plt.imshow(s,cmap='gray')
# plt.title('Saturation')

# plt.subplot(133)
# plt.imshow(v,cmap='gray')
# plt.title('Value')

# plt.show()


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


