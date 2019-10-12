import pdifunFixed as pdi
from matplotlib import pyplot as plt
import cv2
import numpy as np

class TP7:
    """GUIA PRACTICA N7 - NOCIONES DE SEGMENTACION"""

    __basePath = "img/TP7 - Segmentacion/"
    
    def induccion(self):
        
        img = cv2.imread(self.__basePath+"mosquito.jpg", 0)

        self.lineDetection(img)
    


    def ejercicio1(self):
        img = cv2.imread(self.__basePath+"mosquito.jpg", 0)
        noise = pdi.gr_gaussiano(img,0,50)
        imgWithNoise = img + noise
        plt.figure("Ruido")
        plt.subplot(121)
        plt.imshow(img,cmap='gray')
        plt.subplot(122)
        imgWithNoise = cv2.normalize(imgWithNoise, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.imshow(imgWithNoise,cmap='gray')
        plt.show()
        # self.edgeDetectionAlls(img)
        
        self.edgeDetectionAlls(imgWithNoise)



    def lineDetection(self,img):
        grados = 0
        result = pdi.deteccionLineas(img,5, grados)
        val ,result = cv2.threshold(result,250,255,cv2.THRESH_BINARY)

        plt.figure("Deteccion de lineas")
        plt.subplot(121),plt.title("Imagen original")
        plt.imshow(img,cmap='gray')
        plt.subplot(122),plt.title("Lineas de {} grados".format(grados))
        plt.imshow(result,cmap='gray')

        plt.show()

    def pointDetection(self,img):
        
        result = pdi.deteccionPuntos(img)
        val ,result = cv2.threshold(result,200,255,cv2.THRESH_BINARY)        
        plt.figure("Deteccion de puntos")
        plt.subplot(121),plt.title("Imagen original")
        plt.imshow(img,cmap='gray')
        plt.subplot(122),plt.title("Puntos")
        plt.imshow(result,cmap='gray')

        plt.show()

    def edgeDetectionAlls(self, img):

        Gx,Gy = pdi.bordesG_1_derivada(img)
        bPriDerivada = Gx+Gy
        Gx,Gy = pdi.bordesG_Roberts(img)
        bRoberts = Gx+Gy
        Gx,Gy = pdi.bordesG_Prewitt(img)
        bPrewitt = Gx+Gy
        Gx,Gy = pdi.bordesG_Sobel(img)
        bSobel = Gx+Gy
        Gxy   = pdi.bordes_Lapla(img)
        bLapla = Gxy
        Gxy   = pdi.bordes_LoG(img)
        bLog = Gxy


        plt.figure("Edge Detection")
        plt.subplot(231),plt.title("Primera Derivada")
        plt.imshow(bPriDerivada,cmap='gray')

        plt.subplot(232),plt.title("Roberts")
        plt.imshow(bRoberts,cmap='gray')

        plt.subplot(233),plt.title("Prewitt ")
        plt.imshow(bPrewitt,cmap='gray')

        plt.subplot(234),plt.title("Sobel")
        plt.imshow(bSobel,cmap='gray')

        plt.subplot(235),plt.title("Lapla ")
        plt.imshow(bLapla,cmap='gray')

        plt.subplot(236),plt.title("LoG ")
        plt.imshow(bLog,cmap='gray')


        plt.show()


if __name__ == '__main__':

    tp7 = TP7()

    # tp7.ejercicio1()
    tp7.induccion()







    # img = cv2.imread('img/snowman.png',0)
    # img = noisy("gauss",img)
    # plt.subplot(121)
    # plt.imshow(hough_Transform(img,50,13,16), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with LoG')

    # plt.subplot(122)
    # plt.imshow(img, interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with LoG')

    # plt.show()


    


    # # Make plot with vertical (default) colorbar
    # fig, ax = plt.subplots()
    #
    # # plt.subplot(131),plt.imshow(img, cmap = 'gray')
    # # plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(131)
    # plt.imshow(bordes_Prewitt(img), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with Prewitt')
    #
    # plt.subplot(132)
    # plt.imshow(bordes_Sobel(img), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with Sobel')
    #
    # plt.subplot(133)
    # plt.imshow(bordes_Roberts(img), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with Roberts')
    #
    #
    # plt.figure(2)
    #
    # plt.subplot()
    # plt.imshow(bordes_Lapla(img), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with Lapla')
    #
    # plt.figure(3)
    #
    # plt.subplot()
    # plt.imshow(bordes_LoG(img), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with LoG')
    #
    # plt.show()
    # # Add colorbar, make sure to specify tick locations to match desired ticklabels
    # # cbar = fig.colorbar(cax)
    #
    #
    # plt.show()
