# TRABAJO PRACTICO Nro 8 - Morfologia Binaria
# Docs OpenCV
# --------------------------
# https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html
import cv2 as opencv
import numpy as np
from  pdifunFixed import *
from matplotlib import pyplot as plt
import matplotlib.ticker as plticker

class Morfologia :
    PATH = "img/Morfologia/"

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

    def ejercicio1(self, inciso=0):
        """
                   Ejercicio 1 
        ---------------------------------
        Inciso 3: 
        
        """
        
        img = cv2.imread("img/morfologia_ej1.jpg", 0)

        for ielement in range( len(self.SEs)):

            kernel = self.SEs[ielement]
            try:
                img_erosion = cv2.erode(img, kernel, iterations=1)
                img_dilation = cv2.dilate(img, kernel, iterations=1)
            except:
                print "Error with "+ self.nameSES[ielement]
                continue

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
        img = cv2.imread(self.PATH+"matrix_ej2.jpg", 0)
        plt.subplot(131),plt.imshow(img,cmap="gray"),plt.title("Imagen original")
        plt.subplot(132),plt.imshow(opencv.morphologyEx(img,cv.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))  ,cmap="gray"),plt.title("Apertura")
        plt.subplot(133),plt.imshow(opencv.morphologyEx(img,cv.MORPH_CLOSE,self.E1),cmap="gray"),plt.title("Cerradura")
        plt.show()
    
    def ejercicio3_1(self):
        # Kernel 
        kernel = opencv.getStructuringElement(opencv.MORPH_RECT,(2,2) )
        # Get original image
        img = opencv.imread(self.PATH+"Tarjeta.jpeg",0)
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

    def toBinary(self, A ,threshold=127):
        # A = np.asarray(A,dtype=np.int)
        # return opencv.threshold(A,thresh=threshold, maxval=1)
        # retval, threshold = cv2.threshold(A, 127, 255, cv2.THRESH_BINARY)
        # retval, threshold = opencv.threshold(A,threshold,  255, opencv.THRESH_BINARY)
        print threshold

    def invertColor(self, img):
        rows,cols = img.shape
        copyImage = img
        # print img.shape
        for i in range(rows):
            for j in range(cols):
                copyImage[i,j] = -img[i,j]+255
        
        return copyImage

    def extractionConnectedComponnet(self, seed , mask, SE=[]):
        """
        Extraction of connected components
        **********************************
        seed: Matrix (NxM) that contains almost a pixel in the component
        mask: Matrix (NxM) that contains the original figure, and is the mask when aplicating growing (dilation)
        SE: 
        X_k = [X_(k-1) (+) B] \interseccion 

        """

        kernel = SE
        if (kernel == []):
            kernel = opencv.getStructuringElement(opencv.MORPH_RECT,(3,3))
        X_k_1 = seed
        
        result = opencv.dilate(X_k_1,kernel)
        X_k = opencv.bitwise_and(result,mask)
        k_max = 1000
        k = 0
        
        while(not(np.array_equal(X_k,X_k_1)) and k < k_max):
            X_k_1 = X_k
            k+=1
            result = opencv.dilate(X_k_1,kernel)
            X_k = opencv.bitwise_and(result,mask)
        print "extractionConnectedComponnet: INFO - Return with " + str(k)+ " iterations."
        return X_k

    def ejercicio3_2(self):
        x = 13
        y = 31
        # Kernel 
        kernel = opencv.getStructuringElement(opencv.MORPH_RECT,(2,2) )

        img = opencv.imread(self.PATH+"Caracteres.jpeg",0)


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
        masKernel = opencv.getStructuringElement(opencv.MORPH_ELLIPSE,(10,10) )
        

        img = opencv.imread(self.PATH+"estrellas.jpg",0)
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
        kernel = opencv.getStructuringElement(opencv.MORPH_ELLIPSE,(2,2) )
        kernelDetectionStar = np.zeros((size,size),np.uint8)
        for i in range(size):
            kernelDetectionStar[i,i]=1
        kernelDetectionStar = np.rot90(kernelDetectionStar,1)
        
        img = opencv.imread(self.PATH+"lluviaEstrellas.jpg",0)
        plt.figure("Orig")        
        plt.imshow(img,cmap="gray"), plt.title("Original")

        retval, threshold = cv2.threshold(img,125,255,cv2.THRESH_BINARY)
        seed = cv2.erode(threshold,kernelDetectionStar)
        
        result = self.extractionConnectedComponnet(seed,threshold)
        result = cv2.dilate(result,kernel,iterations=8)
        # result = cv2.dilate(result,kernel,iterations=15)
        result = cv2.bitwise_and(img,result)
        plt.figure("Algoritmeada")
        plt.subplot(131) , plt.imshow(seed,cmap="gray"), plt.title("Seed")
        plt.subplot(132) , plt.imshow(threshold,cmap="gray"), plt.title("Threshold")
        plt.subplot(133) , plt.imshow(result,cmap="gray"), plt.title("Result")

        plt.show()

    def frame(self, image,frame_width):
        try:
            rows,cols ,_ = image.shape
            frame_width = 2

            I = np.zeros((rows,cols),np.uint8)

            I[1:1+frame_width,:] = np.ones((frame_width,cols),np.uint8)
            I[rows-frame_width:rows ,:] = np.ones((frame_width,cols),np.uint8)


            I[:,1:1+frame_width] = np.ones((rows,frame_width),np.uint8)
            I[:  ,cols-frame_width:cols] = np.ones((rows,frame_width),np.uint8)

            return I
        except:
            print "Error in frame [function]"

    def morphologicalReconstructionbyDilation(self,F,G,se,k):
        if (k == 0):
            return F
        if (k == 1):
            dilation_F_and_SE = opencv.dilate(F,se) 
            R_D_1 =   opencv.bitwise_and( dilation_F_and_SE, G)
            return R_D_1
        
        RDk_1 =  self.morphologicalReconstructionbyDilation(F,G,se,k-1)
        dilation_RDK1_and_SE = opencv.dilate(RDk_1,se) 
        RDk =   opencv.bitwise_and( dilation_RDK1_and_SE, G)
        if (np.array_equal(RDk,RDk_1)):
            
            print "MR by Dilation finished with ", k , " iterations."
            
            
            return RDk

        else:
            return RDk

    def morphologicalReconstructionbyErotion(self,F,G,se,k):
        if (k == 0):
            return F
        if (k == 1):
            erotion_F_and_SE = opencv.erotion(F,se) 
            RD1 =   opencv.bitwise_or( erotion_F_and_SE, G)
            return RD1
        
        RDk_1 =  self.morphologicalReconstructionbyErotion(F,G,se,k-1)
        erotion_RDK1_and_SE = opencv.erotion(RDk_1,se) 
        RDk =   opencv.bitwise_or( erotion_RDK1_and_SE, G)
        if (np.array_equal(RDk,RDk_1)):
            
            print "MR by Erotion finished with ", k , " iterations."
            
            
            return RDk

        else:
            return RDk

    def difference(self,A,B):
        if (A.shape == B.shape):
            
            rows, cols= A.shape
            result = np.zeros((rows,cols),np.uint8)
            for  i in range(rows):
                for j in range(cols):
                    if (A[i,j] and not(B[i,j])):
                        result[i,j]=1
            return result
    def borderClearing(self,image,mask):

        I = self.frame(image,1)


        
        se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        
        Reconstruction_by_dilation = self.morphologicalReconstructionbyDilation(I,mask,se,40)
        
        
        result = self.difference(mask,Reconstruction_by_dilation)

        return result

    def ejercicio3_5(self):
        img = opencv.imread(self.PATH+"Globulos Rojos.jpg")
        img = cv2.cvtColor(img,opencv.COLOR_BGR2HSV)


     
        # define range of red color in HSV
        lower_color = np.array([0,150,50])
        upper_color = np.array([10,255,255])
        # Threshold the HSV image to get only blue colors
        G = cv2.inRange(img, lower_color, upper_color)
        result = self.borderClearing(img,G)

        # --------------Graphics--------------
        plt.figure("Original")
        plt.subplot(111) , plt.imshow(cv2.cvtColor(img,opencv.COLOR_HSV2RGB),cmap="hsv"), plt.title("The original")

        plt.figure("Algoritmeada")
        plt.subplot(121) , plt.imshow(G,cmap="gray"), plt.title("Mask")
        plt.subplot(122) , plt.imshow(result,cmap="gray"), plt.title("Border Clearing")

        plt.show()


    def normalize(self,img,threshold = 255):
        copyresult = img
        rows,cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if (copyresult[i,j]>240):
                    copyresult[i,j] = 255
        return copyresult

    def convexHull(self,img):
        
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


        # create hull array for convex hull points
        hull = []

        # calculate points for each contour
        for i in range(len(contours)):
            # creating convex hull object for each contour
            hull.append(cv2.convexHull(contours[i], False))

        # create an empty black image
        drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        
        # draw contours and hull points
        for i in range(len(contours)):
            color_contours = (0, 255, 0) # green - color for contours
            color = (255, 0, 0) # blue - color for convex hull
            # draw ith contour
            # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
            # draw ith convex hull object
            cv2.drawContours(drawing, hull, i, color, 1, 8)
        
        return drawing , cv2.fillPoly(drawing, pts =hull, color=(255,255,255))



    def ejercicio3_7(self):
        img = opencv.imread(self.PATH+"Melanoma.jpg")
        img = cv2.cvtColor(img,opencv.COLOR_BGR2HSV)


     
        # define range of red color in HSV
        lower_color = np.array([0,150,50])
        upper_color = np.array([50,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(img, lower_color, upper_color)
        mask = opencv.morphologyEx(mask,opencv.MORPH_CLOSE,opencv.getStructuringElement(opencv.MORPH_RECT,(15,15)))
        convexHull_contours, shape = self.convexHull(np.copy(mask))
        
            
        # --------------Graphics--------------
        plt.figure("Original")
        plt.subplot(111) , plt.imshow(cv2.cvtColor(img,opencv.COLOR_HSV2RGB),cmap="hsv"), plt.title("The original")

        plt.figure("Algoritmeada")
        plt.subplot(121) , plt.imshow(mask,cmap="gray"), plt.title("Mask")
        plt.subplot(122) , plt.imshow(shape), plt.title("Result")

        plt.show()

    def skeleton(self,img):
                
        kSize = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kSize,kSize))
        size = np.size(img)
        skeleton = np.zeros(img.shape,np.uint8)
        
        
        
        done = False

        k = 0
        plt.ion()
        while( not done):
            # First term (A (-) B)
            eroded = cv2.erode(img,kernel)
            # Second term (A (-) B ) o B
            eroded_opening = cv2.morphologyEx(eroded,cv2.MORPH_OPEN,kernel)
            # temp = cv2.dilate(eroded,kernel) 
            # Skeleton of iteration k -> Sk(A)
            # (A (-) B) - (A (-) B ) o B
            Sk = cv2.subtract(eroded,eroded_opening)
            # S(A) = U Sk(A)
            skeleton = cv2.bitwise_or(skeleton,Sk)

            # --- Interactive graphics
            plt.figure(0)
            plt.subplot(131),plt.imshow(eroded,cmap="gray"),plt.title("Eroded")
            plt.subplot(132),plt.imshow(eroded_opening,cmap="gray"),plt.title("Eroded and Opening")
            plt.subplot(133),plt.imshow(Sk,cmap="gray"),plt.title("Skeleton k ")
            plt.figure(5)
            plt.subplot(111),plt.imshow(skeleton,cmap="gray"),plt.title("Skeleton k = "  + str(k))
            
            plt.pause(3)
            plt.clf()

            img = eroded.copy()
                    
            zeros = size - cv2.countNonZero(eroded)
            if zeros==size:
                done = True
            k +=1
            
                
        return skeleton

    def ejercicio3_8(self):
        import time
        image = opencv.imread(self.PATH+"Cuerpos.jpg",0)
        img = np.copy(image)

        size = np.size(img)
        skel = np.zeros(img.shape,np.uint8)
        
        ret,img = cv2.threshold(img,240,255,opencv.THRESH_BINARY_INV)
        kSize = 7
        element = cv2.getStructuringElement(cv2.MORPH_RECT,(kSize,kSize))
        img = opencv.morphologyEx(img,opencv.MORPH_OPEN,element)
        skel = self.skeleton(img)
        

        # plt.figure("Algoritmeada")
        # plt.subplot(121) , plt.imshow(img,cmap="gray"), plt.title("Original")
        # plt.subplot(122) , plt.imshow(skel,cmap="gray"), plt.title("Skeleton")

        # plt.show()
morfologia = Morfologia()

# morfologia.ejercicio1()
# morfologia.ejercicio2()
# morfologia.ejercicio3_1()
# morfologia.ejercicio3_2()
# morfologia.ejercicio3_3()
# morfologia.ejercicio3_4()
# morfologia.ejercicio3_5()
# morfologia.ejercicio3_6() #No implemented, 'cause segmentation is needed
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