import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import pdifunFixed as pdi


# img = cv2.imread('img/huang1.jpg',0)
# plt.figure("Original Image")
# plt.imshow(img,cmap="gray")
# plt.figure("Histogram")
# histogram = cv2.calcHist([img],[0],None,[256],[0,256])
# print "Mean of intensity: " , np.mean(histogram)
# plt.plot(histogram)

# #Equalitationetion
# # img = cv2.imread('img/prueba_EQ.jpg',0)
# #
# plt.figure("Equalitation")
# equ = cv2.equalizeHist(img)
# plt.imshow(equ, cmap='gray')
# histogram_equ = cv2.calcHist([equ],[0],None,[256],[0,256])
# print "Mean of intensity: " , np.mean(histogram)
# plt.figure("Histogram eq")
# plt.plot(histogram_equ)
# plt.show()

# img = cv2.imread('img/imagenD.tif',0)
# plt.figure("Original Image")
# plt.imshow(img,cmap="gray")
# plt.figure("Histogram")
# histogram = cv2.calcHist([img],[0],None,[256],[0,256])
# plt.plot(histogram)
# plt.show()

# ImgA -> Histo 2 CORREEECTAAA
# ImgB -> Histo 4 tomaaaaa
# ImgC -> Histo 1 buenaaa
# ImgD -> Histo 5 cappooo
# ImgE -> Histo 3 mostrooo
#
# res = np.hstack((img,equ)) #stacking images side-by-side
#
#
# plt.subplot(221),plt.imshow(img,cmap='gray'),plt.title("Imagen Original")
# plt.subplot(222),plt.hist(img.ravel(),256,[0,256])
#

# equ =cv2.equalizeHist(equ)
#
# plt.subplot(223),plt.imshow(equ,cmap='gray'),plt.title("Imagen Original")
# plt.subplot(224),plt.hist(equ.ravel(),256,[0,256])
# plt.show()

# Histo1 -> ImgC
# Histo2 -> ImgA
# Histo3 -> ImgE
# Histo4 -> ImgD  | Es la imagen B
# Histo5 -> ImgB   | Es la imagen D



# Ejercicio 3: Filtros pasa-altos

# img = cv2.imread('img/moon.jpg')
# kernel = np.array([ [ 0,  1,  0],
#                     [ 1, -4,  1],
#                     [ 0,  1,  0]
#                     ]
#                   )
# PB = cv2.filter2D(img,-1,-1*kernel)

# plt.subplot(131),plt.imshow(img),plt.title("Original")
# plt.subplot(132),plt.imshow(PB),plt.title("Laplaciano")
# plt.subplot(133),plt.imshow(PB+img),plt.title("Add")
# plt.show()

class TP3:
    """
    Lineal Uniform Transformations
                -
      Histogram | Equalitation
                -
             Filters
    """    
    def __init__(self):
        self.__BASEPATH = "img\TP3 - Operaciones Espaciales"
        
    
    # def lowPassFilter()
    def ejercicio2(self):
        # Ejercicio 2 - Filtros Pasa-bajos

        # void cv::filter2D	(	InputArray 	src,
        #                       OutputArray 	dst,
        #                       int 	ddepth,
        #                       InputArray 	kernel,
        #                       Point 	anchor = Point(-1,-1),
        #                       double 	delta = 0,
        #                       int 	borderType = BORDER_DEFAULT
        # )


        # img = cv2.imread('img\TP3\ejemploLibro.JPG')
     
        
        # kernel3x3 = np.ones((5,5),np.float32)/25
        # kernel5x5 = np.ones((15,15),np.float32)/225
        # kernel7x7 = np.ones((30,30),np.float32)/900
        
        # dst3 = cv2.filter2D(img,-1,kernel3x3)
        # dst5 = cv2.filter2D(img,-1,kernel5x5)
        # # dst7 = cv2.filter2D(img,-1,kernel7x7)
        # sigma = 0.10
        # dst7 = cv2.cv2.GaussianBlur(img,(15,15),sigma)
        
        # plt.subplot(221),plt.imshow(img),plt.title('Original')
        # plt.xticks([]), plt.yticks([])
        
        # plt.subplot(222),plt.imshow(dst3),plt.title('Averaging 5x5')
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(223),plt.imshow(dst5),plt.title('Averaging 15x15')
        # plt.xticks([]), plt.yticks([])
        # plt.subplot(224),plt.imshow(dst7),plt.title('Gaussian 15x15 s='+str(sigma))
        # plt.xticks([]), plt.yticks([])
        # plt.show()

        # -------- Inciso 3 --------
        constelacion = cv2.imread(self.__BASEPATH + "\hubble.tif",0)
        plt.figure(1)
        plt.imshow(constelacion,cmap="gray")
        plt.figure(2)
        imgBlurred = cv2.GaussianBlur(constelacion,(9,9),20)
        val , imgBlurredThresh = cv2.threshold(imgBlurred,150,255,cv2.THRESH_BINARY)
        plt.imshow(imgBlurredThresh,cmap="gray")

        plt.show()
    


    def ejercicio3(self):
        kernel =-1* np.ones((3,3),np.float32)
        
        kernel[1,1]=9

        img = cv2.imread(self.__BASEPATH + "/hubble.tif",0)

        plt.figure("Imagen original")
        plt.imshow(img,cmap='gray')
        plt.figure("Imgen Pasa Alto ")
        
        plt.subplot(121)
        plt.title("Suma 1")
        result = cv2.filter2D(img,0,kernel)
        plt.imshow(result,cmap='gray')  
        plt.subplot(122)      
        kernel[1,1]=8
        plt.title("Suma 0")
        result = cv2.filter2D(img,0,kernel)
        plt.imshow(result,cmap='gray')

        plt.show()

    def diffuseMask(self,img):
        "Unsharp Masking"
        ksize = 5
        kernel =np.ones((ksize,ksize),np.float32)/(ksize*ksize)
        PB = cv2.filter2D(img,0,kernel)
        result = img - PB
        return result

    def highBoost(self,img,k):
        "Highboost filtering"        
        result = pdi.sumaIMG([img , k*self.diffuseMask(img)])
        return result

    def equalizationHistogram(self, img):

        imgEqu = img.copy()
        sumh = []
        sumTotal = 0
        M,N = img.shape[:2]
        #Segun ecuacion 3.3-8 - pag 126 - Rafael Gonzalez
        nj = cv2.calcHist([img],[0],None,[256],[0,256])

        for i in range(len(nj)):
            sumTotal += nj[i][0]
            sumh.append(sumTotal) 
        
        constante = float(float(256-1)/float(M*N))
        sumh = map(lambda x: x * constante,sumh)


        for i in range(M):
            for j in range(N):
                newColor = float(sumh[img[i,j]])
                if newColor > 255:
                    newColor = 255
                imgEqu[i,j] = newColor

        return imgEqu, sumh

    # def stadistics(img):
    #     imgEqu = img.copy()
    #     sumh = []
    #     sumTotal = 0
    #     M,N = img.shape[:2]
    #     histogram = cv2.calcHist([img],[0],None,[256],[0,256])
    #     mean = 0.0

    #     rango = range(len(histogram)
        
    #     for i in rango :
    #         mean += i*histogram[i]

    #     mean = float(mean)/float(M*N)
    #     deviation = 0.0
        
    #     for i in rango:
    #         deviation += (i*histogram[i]-mean)^2
        
    #     deviation = float(deviation)/float(M*N)
        
    #     return mean, deviation

    def ejercicio4(self):
        img = cv2.imread(self.__BASEPATH + "/hubble.tif",0)

        plt.figure("Imagen original")
        plt.imshow(img,cmap='gray')
        plt.figure("Imgen Pasa Bajo ")
        
        plt.subplot(121)
        plt.title("PB")
        PB = self.diffuseMask(img)        
        plt.imshow(PB,cmap='gray')  
        plt.subplot(122)      
              
        plt.title("HighBoost")   
        result =   self.highBoost(img,2)
        plt.imshow(PB+img,cmap='gray')
        plt.show()
    def localEqualizationHistogram(self,img,ksize):
        
        k = ksize//2
        
        imgLocalEquHis = img.copy()

        M,N = img.shape[:2]
        for i in range(M):
            for j in range(N):
                leftInd  = i-k
                rightInd = i +k
                upInd = j-k
                buttomInd = j+k

                if upInd < 0:
                    upInd = 0
                if leftInd < 0:
                    leftInd = 0
                if rightInd > N:
                    rightInd = N
                if buttomInd > M:
                    buttomInd = M
                
                kernelToEqu = img[upInd: buttomInd, leftInd : rightInd]
                localHist = cv2.calcHist([kernelToEqu],[0],None,[256],[0,256])
                
                imgLocalEquHis[i,j] = localHist[img[i,j]]



        return imgLocalEquHis

    def ejercicio5(self):
        img = cv2.imread(self.__BASEPATH+"/cuadros.tif",0)
        result = self.localEqualizationHistogram(img,15)
        plt.subplot(121)
        plt.imshow(img,cmap='gray')
        plt.subplot(122)
        plt.imshow(result,cmap='gray')

        plt.show()

    def pruebas(self):
        img  = cv2.imread(self.__BASEPATH+"/imagenA.tif",0)
        
        print self.stadistics(img)
        return 
        plt.subplot(221)
        plt.imshow(img,cmap='gray')

        plt.subplot(223)
        histogram1 = cv2.calcHist([img],[0],None,[256],[0,256])
        plt.stem(range(len(histogram1)),histogram1)
        
        plt.subplot(222)
        result, transfHistogramFunction = self.equalizationHistogram(img)
        plt.imshow(result,cmap='gray')
        plt.subplot(224)
        histogram2 = cv2.calcHist([result],[0],None,[256],[0,256])
        plt.stem(range(len(histogram2)),histogram2)
        # plt.show()

        plt.figure()
        plt.stem(range(len(transfHistogramFunction)),transfHistogramFunction)
        plt.show()
        
        # plt.figure()
        # plt.stem(range(len(sumh)),sumh)
    
        # plt.show()



if __name__ == "__main__":
    print "Estamos en el programa principal : "
    tp3 = TP3()

    # tp3.ejercicio4()
    tp3.ejercicio5()
    # tp3.pruebas()
