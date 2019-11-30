import cv2
import numpy as np
import  pdifunFixed  as pdi
from matplotlib import pyplot as plt
import cmath
import math

# encoding: utf-8
# TRABAJO PRACTICO Nro 6 - Restauracion y Reconstruccion

class TP6:
    __BASEPATH= "img/TP6 - Restauracion y Reconstruccion/"
        
    def filtrosNoLineales(self):
        
        nameImage = self.__BASEPATH +  "ejemploRuido.jpg"        
        img = cv2.imread(nameImage,0)
        img= pdi.noisy("gauss",img)
        plt.subplot(121)
        plt.imshow(img,cmap='gray')
        plt.subplot(122)
        result = self.orderStatistcFilter(img,5,ALNRFilter)
        plt.imshow(result,cmap='gray')

        plt.figure("Histogramas")
        plt.subplot(121)
        histoIMG = pdi.histograma(img)
        plt.stem(range(len(histoIMG)),histoIMG,markerfmt='')
        plt.subplot(122)
        histoIMGFILTERED = pdi.histograma(np.uint8(result))
        plt.stem(range(len(histoIMGFILTERED)),histoIMGFILTERED,markerfmt='')

        plt.show()

    def ejercicio1(self,nombreArchivo=None):
        
        nameFile = "ejemploRuido.jpg" 
        
        if (nombreArchivo != None):        
            nameFile = nombreArchivo

        nameImage = self.__BASEPATH + nameFile     

        img = cv2.imread(nameImage,0)

        # print img.dtypeP
        img_noise = img.copy()
        #Ruidos:
        #   s&p : Sal y Pimienta
        #   s : Sal
        #   p : Pimienta
        #   gauss : Ruido Gaussiano
        #   poisson:  Posion
        #   speckle: ai ron nog


        img_noise = pdi.noisy("gauss",img)
        # img_noise = pdi.gr_salPimienta(img,0,0.1)

        plt.figure("Original")

        plt.subplot(221)
        plt.imshow(img,cmap='gray')
       
        plt.subplot(222)
        plt.imshow(img_noise,cmap='gray')
                
        histo1 = pdi.histograma(img)
        plt.subplot(223)
        plt.title("Hist Img Original")
        plt.stem(range(len(histo1)),histo1,markerfmt='')

        histo2 = pdi.histograma(img_noise)
        plt.subplot(224)
        plt.title("Hist Img con Ruido")
        plt.stem(range(len(histo2)),histo2,markerfmt='')
        
        plt.show()
        plt.figure("Filtros Lineales")
        plt.subplot(231)
        plt.title("Media Aritmetica")
        mediaAritmetica = pdi.media_aritmetica(img_noise,3,3)
        plt.imshow(mediaAritmetica,cmap='gray')
        plt.subplot(232)
        plt.title("Media geometrica")
        mediaGeometrica = pdi.media_geometrica(img_noise,3,3)
        plt.imshow(mediaGeometrica,cmap='gray')
        plt.subplot(233)
        plt.title("Media ContraArmonica")
        mediaContraArmnica = pdi.media_contraArmonica(img_noise,50,3,3)
        plt.imshow(mediaContraArmnica,cmap='gray')
        plt.subplot(234)
        histoMediaAritmetica =  pdi.histograma(mediaAritmetica)
        plt.stem(range(len(histoMediaAritmetica)),histoMediaAritmetica,markerfmt='')
        plt.subplot(235)
        histoMediaGeometrica = pdi.histograma(mediaGeometrica)
        plt.stem(range(len(histoMediaGeometrica)),histoMediaGeometrica,markerfmt='')
        plt.subplot(236)
        histoMediaContraArmnica = pdi.histograma(mediaContraArmnica)
        plt.stem(range(len(histoMediaContraArmnica)),histoMediaContraArmnica,markerfmt='')


        plt.figure("Filtros No Lineales")
        plt.subplot(231)
        plt.title("Mediana")
        mediaAritmetica = pdi.orderStatistcFilter(img_noise,5,np.median)
        plt.imshow(mediaAritmetica,cmap='gray')
        plt.subplot(232)
        plt.title("Alfa recortado")
        mediaGeometrica = pdi.orderStatistcFilter(img_noise,5,alphaTrimmedFilter)
        plt.imshow(mediaGeometrica,cmap='gray')
        plt.subplot(233)
        plt.title("Punto medio")
        mediaContraArmnica = pdi.orderStatistcFilter(img_noise,5,midpointFilter)
        plt.imshow(mediaContraArmnica,cmap='gray')
        plt.subplot(234)
        histoMediaAritmetica =  pdi.histograma(np.uint8(mediaAritmetica))
        plt.stem(range(len(histoMediaAritmetica)),histoMediaAritmetica,markerfmt='')
        plt.subplot(235)
        histoMediaGeometrica = pdi.histograma(np.uint8(mediaGeometrica))
        plt.stem(range(len(histoMediaGeometrica)),histoMediaGeometrica,markerfmt='')
        plt.subplot(236)
        histoMediaContraArmnica = pdi.histograma(np.uint8(mediaContraArmnica))
        plt.stem(range(len(histoMediaContraArmnica)),histoMediaContraArmnica,markerfmt='')

        plt.show()


        # plt.hist(img_noise.ravel(),256,[0,256]); plt.show()



        # cv2.waitKey(0)
    
    def ejercicio5(self):
        "Filtrado Inverso"

        nameImg = "ejemploRuido.jpg"
        img = cv2.imread(self.__BASEPATH+nameImg,0)
        M,N = img.shape[:2]
 
        # H = Hmovida(M,N,0.1,0.1)
        H = pdi.filtro_butterworth(M,N,20,2)
        
        plt.figure("Imagen y H")
        imgDegradada = pdi.filtro_img(img,H)

        plt.figure("Imagen")
        plt.imshow(imgDegradada,cmap='gray')
        

        # #Solucion 1: Pseudo-Inverso
        R1 = inverseFiltering(H)
        # R3 = 1/H
        # #Solucion 2: Suavizar la relacion con un PasaBajos
        # # R2 = inverseFiltering(H,pseudo=False)
        # Do = 10
        # filtroPB = pdi.filtro_gaussiano(M,N,Do)
        # R2 = cv2.multiply(R3,filtroPB)
        # plt.figure("filtro gausiano")
        # plt.imshow(filtroPB,cmap='gray')
        # plt.show()
        plt.figure("Imagen Degradada y R")
        imgRestaurada = pdi.filtro_img(imgDegradada,R1)
        plt.imshow(imgRestaurada,cmap='gray')
        plt.show()

    def ejercicio6(self):
        #
        # nameImg = "FAMILIA_a.jpg"     #Gausiano o=10 Alfa recortado / Armonica
        # nameImg = "FAMILIA_b.jpg"       #Uniforme  Geometrica/Armonica
        # nameImg = "FAMILIA_c.jpg"     #Sal and Pepper 5% Mediana
        nameImg = "FAMILIA.jpg"     #Original

        img = cv2.imread(self.__BASEPATH+nameImg,0)



        # -------- Estimacion de los parametros del ruido -------- 
        # histo = pdi.infoROI(img)
        # stadisticsHisto(histo)


        plt.figure("Original")

        plt.subplot(121)
        plt.imshow(img,cmap='gray')
        histo1 = pdi.histograma(img)
        plt.subplot(122)
        plt.title("Hist Img Original")
        plt.stem(range(len(histo1)),histo1,markerfmt='')


        
        plt.figure("Filtros Lineales")
        plt.subplot(121)
        # print "Media Aritmetica"
        plt.title("Media Aritmetica")               
        # filtrado = pdi.media_aritmetica(img,3,3)
        # filtrado = pdi.media_geometrica(img,5,5)
        # filtrado = pdi.media_armonica(img,3,3)
        # filtrado = pdi.media_contraArmonica(img,3,3,-2)
        # func = np.median
        # func = pdi.midpointFilter
        # func = pdi.ALNRFilter
        func = pdi.alphaTrimmedFilter
        filtrado = pdi.orderStatistcFilter(img,5,func)
        plt.title("Imagen filtrada")
        plt.imshow(filtrado,cmap='gray')
        plt.subplot(122)
        filtradoC = np.uint8(filtrado)
        histoFiltro = pdi.histograma(filtradoC)
        plt.stem(range(len(histoFiltro)),histoFiltro,markerfmt='')
        plt.show()


def stadisticsHisto(histo):

    mean = 0
    variance = 0
    MN = 0
    hN = len(histo)
    
    for z in range(hN):
        MN += histo[z]

    print "Size: ", MN

    for z in range(hN):
        mean += histo[z]*z

    mean = mean/MN

    print "Mean: ", mean

    for z in range(hN):
        variance += histo[z]*(z-mean)**2
    
    variance = variance/MN

    print "Desviation : ", math.sqrt(variance)

    return float(mean), float(variance)

def inverseFiltering(H):
    R = np.zeros(H.shape)
    e= 0.1 
    for i in range(H.shape[0]):
        for j in range(H.shape[0]):
            if np.abs(H[i,j]) > e:
                R[i,j] = 1/H[i,j]
            else:
                R[i,j] = 0                
    return R
            
def inverseFdailtering(H):
    R = np.zeros(H.shape)
    e= 0.1
    for i in range(H.shape[0]):
        for j in range(H.shape[0]):
            if np.abs(H[i,j]) > e:
                R[i,j] = 1/H[i,j]
            else:
                R[i,j] = 0                
    return R

def Hmovida(M,N,a,b):
    H = np.zeros((M,N))


    for u in range(M):
        for v in range(N):
            
            arg = math.pi*(u*a+v*b)
            if arg == 0:
                arg =0.01
            ftr = (1/arg )
            seno = math.sin(arg)
            exponencial = cmath.exp(arg*0.j)
            H[u,v] = 1 * seno * 1
            # H[u,v]// =nc.real  
            # H[u,v,1] =nc.imag  


    return H


if __name__=="__main__":
    
    tp6 = TP6()

    # tp6.ejercicio1()
    # tp6.ejercicio4()
    tp6.ejercicio5()
    # tp6.ejercicio6()
    # tp6.filtrosNoLineales()

    # l = [[12.0, 0.0], [13.0, 2.0], [15.0, 3.0], [17.0, 3.0], [2.0, 14.0], [2.0, 17.0], [0.0, 20.0]]
    


    # def filterImg(img, filtro_magnitud):
    # """Filtro para imgenes de un canal"""

    # # como la fase del filtro es 0 la conversin de polar a cartesiano es directa (magnitud->x, fase->y)
    # filtro = np.array([filtro_magnitud, np.zeros(filtro_magnitud.shape)]).swapaxes(0, 2).swapaxes(0, 1)
    # imgf = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # imgf = cv2.mulSpectrums(imgf, np.float32(filtro), cv2.DFT_ROWS)

    # return cv2.idft(imgf, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)