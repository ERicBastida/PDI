  # REPASO GUIAS PRACTICAS 2018
    # OPERACIONES PUNTUALES

import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


class TP1:
    "Transformaciones Lineales"

    def __init__(self):
        self.LUT  = range(256)
        self.LogT = range(256)
        self.PowT = range(256)


    def showLUT(self):
        return self.LUT

    def changeLUT(self,a=1,c=0):

        for li  in range(len(self.LUT)):
            value = a*li + c
            if value > 255:
                value = 255
            elif value < 0 :
                value = 0
            self.LUT[li]= value

    def changeLogT(self,c=1):

        for li  in range(len(self.LogT)):
            value = c*math.log(1+li)
            if value > 255:
                value = 255
            elif value < 0 :
                value = 0
            self.LogT[li]= value            
            
    def changePowT(self,c=1):

        for li  in range(len(self.PowT)):
            value = c*math.pow(li,c)
            if value > 255:
                value = 255
            elif value < 0 :
                value = 0
            self.PowT[li]= value      

    def update_image(self, img_org, option = 1):
        img = img_org.copy()
        
        rows, cols = img.shape[:2]
        for r in range(rows):
            for c in range(cols):
                if   option == 1:
                    img[r, c] = self.LUT[img[r, c]]
                elif option == 2:
                    img[r, c] = self.LogT[img[r, c]]
                elif option == 3:
                    img[r, c] = self.PowT[img[r, c]]

        return img


    def inciso1(self,pathIMG,nameTitle="Resultado"):

        try:
            
            img = plt.imread(pathIMG,0)

            plt.subplot(211)
            plt.title("Original")
            plt.imshow(img,cmap='gray')
            plt.subplot(212)
            plt.title(nameTitle)
            img_result = self.update_image(img)
            plt.imshow(img_result,cmap='gray')

            plt.show()
        except Exception as e:
            print "Ha ocurrido un error de tipo :" + str(e)

    def inciso2(self,pathIMG,nameTitle="Resultado"):
            try:
            
                img = plt.imread(pathIMG,0)

                plt.subplot(211)
                plt.title("Original")
                plt.imshow(img,cmap='gray')
                plt.subplot(212)
                plt.title(nameTitle)
                img_result = self.update_image(img)
                plt.imshow(img_result,cmap='gray')

                plt.show()
            except Exception as e:
                print "Ha ocurrido un error de tipo :" + str(e)
        

                
    def sumaIMG(self, imgs):
        N = len(imgs)
        acumIMG = np.zeros(imgs[0].shape)
        for img in imgs:
            acumIMG += img 
        acumIMG =  acumIMG/N
        return acumIMG

    def restaIMG(self, imgA,imgB):
        result = np.zeros(imgA.shape)
        if imgA.shape == imgB.shape:
            (rows,cols ) = imgA.shape
            for r in range(rows):
                for c in range(cols):
                    value = imgA[r,c]-imgB[r,c]
                    if value > 0:
                        result[r,c] = value
        return result
    def multiIMG(self,A,mask):
        mask = cv.threshold(A,125,1,cv.THRESH_BINARY_INV)
        #mask = cv.cvtColor(cv.UMat(mask),cv.COLOR_RGB2GRAY)
        print type(mask[1]), " - ", type(A)
        return cv.multiply(A,mask[1])

    def inciso3(self,pathIMG):
            img1 = plt.imread(pathIMG,0)
            img2 = img1.copy()
            img3 = img2.copy()

            listImgs = [img1,img2,img3]

            plt.subplot(211)
            plt.title("Original")
            plt.imshow(img1,cmap='gray')
            plt.subplot(212)
            plt.title("Multiply")
            img_result = self.multiIMG(img1,img2)
            plt.imshow(img_result,cmap='gray')

            plt.show()

    def bitSlicing(self, img ):
        rows, cols = img.shape[:2]
        for i in range(rows):
            for j in range(cols):

                binario = bin(img[i,j])
        
                binario = binario.split('b')[1]
                binario = binario[:-1]
                N = len(binario)
            
                value  = 0
                for i in range(N):
                    value = int(binario[i])*math.pow(2,i) 
                img[i,j] = value
            

        
    def inciso4(self,pathIMG):
        img = plt.imread(pathIMG,0)
        self.bitSlicing(img)


if __name__ == "__main__":
    tp = TP1()
    #a = -1, c = 255
    #tp.changeLUT(a, c)
    #tp.inciso1("img/billete.jpg","a={} , c = {}".format(a,c))

    tp.inciso4("img/billete.jpg")
