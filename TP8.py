
# encoding: utf-8

# TRABAJO PRACTICO Nro 8 - Morfologia Binaria
# Docs OpenCV
# --------------------------
# https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
import cv2 
import numpy as np
import pdifunFixed as pdi
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker

class Morfologia :
    # PATH = "img/Morfologia/"

    # It was created one time, becouse is needed to create and image

    # MATRIXE1 = np.array(
    # [
    #     [0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,1,1,1,0,0,0,0,0,0,0,0],
    #     [0,0,0,1,0,0,0,0,1,0,0,0],
    #     [0,0,0,1,0,0,0,1,1,1,0,0],
    #     [0,0,1,1,1,0,0,1,1,1,0,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0],
    #     [0,1,0,0,0,0,0,0,0,0,0,0],
    #     [0,0,1,0,0,0,0,0,1,1,1,0],
    #     [0,0,0,1,0,0,0,0,0,0,1,0],
    #     [0,0,0,0,0,0,0,1,1,1,1,0],
    #     [0,1,0,0,0,0,0,0,1,1,1,0],
    #     [0,0,0,0,0,0,0,0,0,0,0,0],
    # ]
    # )
    
    MATRIXE2 = np.array(
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0],
        [0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ]
    )
    E1 = np.array(
        [
        [1 , 1 , 1],
        [1 , 1 , 1],
        [1 , 1 , 1]
        ]
    )

    E2 = np.array(
        [
        [0 , 1 , 0],
        [1 , 1 , 1],
        [0 , 1 , 0]
        ]
    )
    # plt.imsave("img/E2.jpg",E1)
    E3 = np.array(
        [
        [0 , 1 , 0],
        [0 , 1 , 1],
        [0 , 1 , 0]
        ]
    )

    E4 = np.array(
        [
        [0 , 1 , 0],
        [0 , 1 , 1],
        [0 , 0,  0]
        ]
    )

    E5 = np.array(
        [
        [0 , 1 , 1],
        [0 , 1 , 1],
        [0 , 1,  1]
        ]
    )

    E6 = np.array(
        [
        [0 , 0 , 1],
        [0 , 0 , 1],
        [0 , 0,  1]
        ]
    )
    K = 500
    SEs = [E1,E2,E3,E4,E5,E6]
    nameSES = ["E1","E2","E3","E4","E5","E6"]

    __basePath = "img/TP8 - Morfologia/"

    def ejercicio1(self, inciso=0):
        """
                   Ejercicio 1 
        ---------------------------------
        Inciso 3: 
        
        """
        
        img =cv2.imread(self.__basePath+"morfologia_ej1.jpg", 0)
        img = np.uint8(img)

        for ielement in range( len(self.SEs)):

            kernel = np.uint8( self.SEs[ielement])


            img_erosion = cv2.erode(img, kernel, iterations=1)
            img_dilation = cv2.dilate(img, kernel, iterations=1)


            plt.figure(self.nameSES[ielement])

            plt.subplot(141)
            plt.title("Original")
        
            plt.imshow(img,cmap="gray")

            plt.subplot(142)
            plt.title("Kernel")
            plt.grid()
            plt.imshow(kernel,cmap="gray")
            

            plt.subplot(143)
            plt.title("Dilatacion")
            plt.grid()
            plt.imshow(img_dilation,cmap="gray")

            plt.subplot(144)
            plt.title("Erosion")
            plt.grid()
            plt.imshow(img_erosion,cmap="gray")
            

        plt.show()

    def ejercicio2(self):

        # img = cv2.imread(self.__basePath+"matrix_ej2.jpg", 0)
        img = self.MATRIXE2
        img = np.uint8(img)
        
        plt.subplot(131),plt.imshow(img,cmap="gray"),plt.title("Imagen original")
        plt.subplot(132),plt.imshow(opencv.morphologyEx(img,cv.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))  ,cmap="gray"),plt.title("Apertura")
        plt.subplot(133),plt.imshow(opencv.morphologyEx(img,cv.MORPH_CLOSE,self.E1),cmap="gray"),plt.title("Cerradura")
        plt.show()
    
    def ejercicio3_1(self):
        # Kernel 
        kernel = opencv.getStructuringElement(opencv.MORPH_RECT,(2,2) )
        # Get original image
        img = opencv.imread(self.__basePath +"Tarjeta.jpeg",0)
        # Threshold image
        retval, threshold = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
        # Inverting colors
        mask = self.invertColor(threshold)
        mask = opencv.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Getting seed to extrac connected components
        seed = opencv.erode(mask,kernel,iterations = 2)

        # seed = self.invertColor(seed)

        # Component Connected
        result = self.extractionConnectedComponnet(seed,mask)


        plt.subplot(131),plt.imshow(seed,cmap="gray"),plt.title("SEED")
        plt.subplot(132),plt.imshow(mask,cmap="gray"),plt.title("MASK")
        plt.subplot(133),plt.imshow(result,cmap="gray"),plt.title("RESULT")
        
        plt.figure("Original")
        plt.imshow(img,cmap="gray")

        
        

        
        # plt.subplot(133),plt.imshow(result,cmap="gray"),plt.title("Algoritmeada")

        plt.show()

    def ejercicio3_2(self):
        x = 13
        y = 31
        # Kernel 
        kernel = opencv.getStructuringElement(opencv.MORPH_RECT,(2,2) )
        pathFile = self.__basePath + "Caracteres.jpeg"
        img = opencv.imread(pathFile,0)
        

        retval, thresholdORG = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
        threshold = self.invertColor(thresholdORG)
        seed = opencv.erode(threshold,kernel,iterations = 2)
        result =  self.extractionConnectedComponnet(seed,threshold)


        plt.subplot(131),plt.imshow(img,cmap="gray"),plt.title("Original Image")
        plt.subplot(132),plt.imshow(threshold,cmap="gray"),plt.title("Mask")
        plt.subplot(133),plt.imshow(result,cmap="gray"),plt.title("Result")
        
        plt.figure()
        invertMask = opencv.dilate(result,kernel,iterations = 3)

        withOutLetters = invertMask +img  
        # withOutLetters = opencv.threshold(withOutLetters,255,255,cv.THRESH_BINARY)

      
        plt.subplot(121) ,plt.imshow(invertMask,cmap="gray"), plt.title("iNVERT MASK")
        plt.subplot(122) ,plt.imshow(self.normalize(withOutLetters),cmap="gray"), plt.title("WithOut letters")

        plt.show()
    
    def ejercicio3_3(self):
        # Kernel 
        kernel = opencv.getStructuringElement(opencv.MORPH_ELLIPSE,(5,5) )
        masKernel = opencv.getStructuringElement(opencv.MORPH_ELLIPSE,(10,10))
        

        img = opencv.imread(self.__basePath+"estrellas.jpg",0)
        plt.figure("Orig")
        retval, threshold = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
        result = cv2.erode(threshold,kernel)
        result = cv2.dilate(result,masKernel,iterations=4)
        result = cv2.bitwise_and(img,result)
        
        plt.imshow(img,cmap="gray"), plt.title("Original")
        plt.figure("Algoritmeada")
        plt.subplot(121) , plt.imshow(threshold,cmap="gray"), plt.title("Threshold")
        plt.subplot(122) , plt.imshow(result,cmap="gray"), plt.title("Result")

        plt.show()

    def ejercicio3_4(self):

        size = 5
        
        kernelDetectionStar = np.zeros((size,size),np.uint8)
        # Genero un kernel donde su diagonal contiene todos unos
        # es decir, obtengo un detector de lineas en 45° (en realidad de -45°)
        for i in range(size):
            kernelDetectionStar[i,i]=1
        # Lo roto para obtenerlo a 45
        kernelDetectionStar = np.rot90(kernelDetectionStar,1)
        
        img = opencv.imread(self.__basePath+"lluviaEstrellas.jpg",0)
        plt.figure("Orig")        
        plt.imshow(img,cmap="gray"), plt.title("Original")

        retval, threshold = cv2.threshold(img,125,255,cv2.THRESH_BINARY)
        seed = cv2.erode(threshold,kernelDetectionStar)
        
        result = self.extractionConnectedComponnet(seed,threshold)

        kernel = opencv.getStructuringElement(opencv.MORPH_ELLIPSE,(2,2) )
        result = cv2.dilate(result,kernel,iterations=8)
        
        result = cv2.bitwise_and(img,result)

        plt.figure("Algoritmeada")
        plt.subplot(131) , plt.imshow(seed,cmap="gray"), plt.title("Seed")
        plt.subplot(132) , plt.imshow(threshold,cmap="gray"), plt.title("Threshold")
        plt.subplot(133) , plt.imshow(result,cmap="gray"), plt.title("Result")

        plt.show()

    def ejercicio3_5(self):
        img = opencv.imread(self.__basePath+"Globulos Rojos.jpg")
        img = cv2.cvtColor(img,opencv.COLOR_BGR2HSV)

        G = segmentador(img,[0,150,50],[10,255,255])
     

        plt.figure()
        
        plt.imshow(G,cmap='gray'),plt.title("Segmentada")
        plt.show()

        result = self.borderClearing(img,G)

        # --------------Graphics--------------
        plt.figure("Original")
        plt.subplot(111) , plt.imshow(cv2.cvtColor(img,opencv.COLOR_HSV2RGB),cmap="hsv"), plt.title("The original")

        plt.figure("Algoritmeada")
        plt.subplot(121) , plt.imshow(G,cmap="gray"), plt.title("Mask")
        plt.subplot(122) , plt.imshow(result,cmap="gray"), plt.title("Border Clearing")

        plt.show()
    
    def ejercicio3_6(self):
        
        img = cv2.imread(self.__basePath+"Rio.jpeg")
        imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # infoROI(imgHSV)

        mask_lago = segmentador(imgHSV,
                            [109, 228, 190],
                            [114, 255, 235]
                            )
        mask_rios = segmentador(imgHSV,
                            [70, 45, 118],
                            [100, 90, 255]
                                    )
        mask = cv2.bitwise_or(mask_lago,mask_rios)             
        result = masking(imgHSV,mask)        
        result = cv2.cvtColor(result,cv2.COLOR_HSV2RGB)              
        # mask = cv2.morphologyEx(mask,cv2.MORPH_ERODE,(10,10))

        plt.figure("Previsualizacion ")
        plt.subplot(121)
        plt.imshow(imgRGB)
        plt.subplot(122)
        plt.imshow(result,cmap="gray")
        plt.show()   

    def ejercicio3_7(self):
        img = cv2.imread(self.__basePath+"Melanoma.jpg")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        mask = segmentador(img,[0,150,50],[50,255,255])
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))

        contornosDibujados, shape = self.convexHull(np.copy(mask))

        plt.figure()
        plt.imshow(contornosDibujados)
        plt.show()
        
            
        # --------------Graphics--------------
        plt.figure("Original")
        plt.subplot(111) , plt.imshow(cv2.cvtColor(img,cv2.COLOR_HSV2RGB),cmap="hsv"), plt.title("The original")

        plt.figure("Algoritmeada")
        plt.subplot(121) , plt.imshow(mask,cmap="gray"), plt.title("Mask")
        plt.subplot(122) , plt.imshow(shape), plt.title("Result")

        plt.show()

    def ejercicio3_8(self):
        # import time
        image = cv2.imread(self.__basePath+"Cuerpos.jpg",0)
        img = np.copy(image)

        size = np.size(img)
        skel = np.zeros(img.shape,np.uint8)
        
        ret,img = cv2.threshold(img,240,255,cv2.THRESH_BINARY_INV)
        kSize = 7
        element = cv2.getStructuringElement(cv2.MORPH_RECT,(kSize,kSize))
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,element)
        skel = self.skeleton(img)
        

        # plt.figure("Algoritmeada")
        # plt.subplot(121) , plt.imshow(img,cmap="gray"), plt.title("Original")
        # plt.subplot(122) , plt.imshow(skel,cmap="gray"), plt.title("Skeleton")

        plt.show()






morfologia = Morfologia()

# morfologia.ejercicio1()
# morfologia.ejercicio2()
# morfologia.ejercicio3_1()
# morfologia.ejercicio3_2()
# morfologia.ejercicio3_3()
# morfologia.ejercicio3_4()
# morfologia.ejercicio3_5()
# morfologia.ejercicio3_6() 
# morfologia.ejercicio3_7() 
morfologia.ejercicio3_8() 





#---------------------------------Docs---------------------------------
#----------------------------Python/OpenCV-----------------------------

#-KERNELS - Estructure Elements

# kernel = np.ones((size,size),np.uint8)
# cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

#-Threshold
# ret,img = cv.threshold(img,127,255,cv.THRESH_BINARY)
         # cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)


#-Morphological Operations
# opening  = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# closing  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
# gradient  = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# top_hat  = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)



# np.rot90(kernelDetectionStar,1)

# Counts non-zero array elements
# cv2.countNonZero(src) 

#-Example of interactive plotting

# plt.ion()
# for i in range(50):
#     y = np.random.random([10,1])
#     plt.plot(y)
#     plt.draw()
#     plt.pause(0.0001)
#     plt.clf()

# ----------------------FIN DOC-------------------------


# def onclick(event):
#     global ix, iy
#     ix, iy = event.xdata, event.ydata
#     print 'x = %d, y = %d'%(
#         ix, iy)

#     global coords
#     coords.append((ix, iy))

#     if len(coords) == 2:
#         fig.canvas.mpl_disconnect(cid)

#     return coords

# img = cv2.imread("img/billete.jpg",0)
# fig = plt.figure()

# print "Holis"

# plt.imshow(img)
# plt.show()


# coords = []

# cid = fig.canvas.mpl_connect('button_press_event', onclick)
