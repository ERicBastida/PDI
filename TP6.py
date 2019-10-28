import cv2
import numpy as np
import  pdifunFixed  as pdi
from matplotlib import pyplot as plt




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

    def ejercicio1(self):

        nameImage = self.__BASEPATH +  "ejemploRuido.jpg"        
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
        mediaGeometrica = pdi.filtroMediaGeometrica(img_noise,3,3)
        plt.imshow(mediaGeometrica,cmap='gray')
        plt.subplot(233)
        plt.title("Media ContraArmonica")
        mediaContraArmnica = pdi.filtroMediaContraarmonica(img_noise,50,3,3)
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

    def ejercicio4(self):
        img = cv2.imread(self.__BASEPATH+"img_degradada.tif",0)
 
        spectrum = pdi.spectrum(img)

        val, spectrum2 = cv2.threshold(spectrum,0.8,1,cv2.THRESH_BINARY)
        # plt.imshow()
        result = np.where(spectrum2 == np.amax(spectrum2))

       
        print('Tuple of arrays returned : ', result)
        
        print('List of coordinates of maximum value in Numpy array : ')
        # zip the 2 arrays to get the exact coordinates
        listOfCordinates = list(zip(result[0], result[1]))
    
        # for cord in listOfCordinates:
        #     print(cord)
        
        print listOfCordinates[2][0] , " - ", listOfCordinates[2][1]

        d = pdi.dist(listOfCordinates[2][0],listOfCordinates[2][1] )


        print np.linalg.norm(np.array(5)-np.array(5))
        print "Distancia : ", d

        M,N = spectrum2.shape[:2]
        D = pdi.dist([0,0],[0.5*M,0.5*N])
        print D , M , N
        Do = float(114/D)

        print "Frecuencia de corte : ", Do
        
        filtro = pdi.filtro_ideal(M,N,10,70,90,False)

        resultado = pdi.filtro_img(img,filtro)

        plt.imshow(resultado,cmap='gray')
        
        plt.show()

if __name__=="__main__":
    
    tp6 = TP6()

    # tp6.ejercicio1()
    tp6.ejercicio4()
    # tp6.filtrosNoLineales()

    
