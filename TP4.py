# GUIA PRACTICA N4 - COLOR


import cv2
import numpy as np
from matplotlib import pyplot as plt
import pdifunFixed as pdi

def pegar(imgA,mask,imgB):
    """
    Recibe dos imagenes A y B con sus respectivos 3 canales
      además de una mascara que sera la que va a recortar 
        la imagen B y la va a pegar sobre la imagen A
    
    """
    # Separo las imagenes en sus respectivos canales
    A1,A2,A3 = cv2.split(imgA)    
    B1,B2,B3 = cv2.split(imgB)


    # Normalizo los canales para realizar la multiplicacion
    A1 = np.uint8(A1)
    A2 = np.uint8(A2)
    A3 = np.uint8(A3)
    mask = np.uint8(mask)
    B1 = np.uint8(B1)
    B2 = np.uint8(B2)
    B3 = np.uint8(B3)

    mask = cv2.normalize(mask, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    C1 = cv2.multiply(mask,B1)
    C2 = cv2.multiply(mask,B2)
    C3 = cv2.multiply(mask,B3)
    
    
    mask_inv = cv2.bitwise_not(mask)
    mask_inv = cv2.normalize(mask_inv, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    D1 = cv2.multiply(mask_inv,A1)
    D2 = cv2.multiply(mask_inv,A2)
    D3 = cv2.multiply(mask_inv,A3)

    E1 = C1 + D1
    E2 = C2 + D2
    E3 = C3 + D3

    result = cv2.merge((E1,E2,E3))

    return result

def segmentadorRGB(img,r):
    M,N = img.shape[:2]

    for i in range(M):
        for j in range(N):
            

def segmentadorHSI(img,deltaH):
    pass

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
    
    def ejercicio2(self):
        img = cv2.imread(self.__basePath+"rio.jpg")

        R,_,_ = cv2.split(img)
        plt.figure()
        plt.imshow(img)
        plt.show()

        I = np.ones(R.shape)

                               # G     B    R
        amarillo  = cv2.merge(( 255*I,255*I,0*I) )

        frame_threshold = cv2.inRange(img, (0, 0, 0), (20, 20, 20))
        # frame_threshold = cv2.bitwise_not(frame_threshold)


        resultado = pegar(img,frame_threshold,amarillo)





        plt.subplot(131)
        plt.imshow(img,cmap='gray')

        # histo = pdi.histograma(img)
        plt.subplot(132)
        plt.imshow(frame_threshold,cmap='gray')

        plt.subplot(133)
        plt.imshow(resultado,cmap='gray')
       
    

        plt.show()

    

tp4 = TP4()

# tp4.ejercicio1()
tp4.ejercicio2()
        



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


