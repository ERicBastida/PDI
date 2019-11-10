import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
basePath = sys.path[0]



if __name__ == '__main__':

    # nameImage = "/B2C1_01.jpg"
    # nameImage = "/B2C1_02a.jpg"
    # nameImage = "/B5C1_01.jpg"
    # nameImage = "/B5C1_02a.jpg"
    # nameImage = "/B10C1_01.jpg"
    # nameImage = "/B10C1_02a.jpg"
    # nameImage = "/B20C1_01a.jpg"
    # nameImage = "/B20C1_02.jpg"
    # nameImage = "/B50C1_01a.jpg"
    # nameImage = "/B50C1_02.jpg"
    # nameImage = "/B100C1_01.jpg"
    nameImage = "/B100C1_02a.jpg"

    img = cv2.imread(basePath+nameImage )

    derecho = cv2.cvtColor( img[:, 0:180], cv2.COLOR_BGR2GRAY)
    _,derecho = cv2.threshold(derecho,175,255,cv2.THRESH_BINARY)
    # plt.imshow(derecho,cmap='gray')    
    # plt.show()

    porcentaje = float((derecho > 0).sum())/(derecho.shape[0]*derecho.shape[1])
    #Si el porcentaje de blancos es demasiado bajo, quiere decir que tomamos el billete al reves
    if porcentaje < 0.75:
        img = pdi.rotar(img,180)
    
    diamantes = cv2.cvtColor( img[0:80, 133:180], cv2.COLOR_BGR2GRAY)
    _, maskDiamantes = cv2.threshold(diamantes,100,255,cv2.THRESH_BINARY_INV)


    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    objetos = pdi.gestionarObjetos(maskDiamantes)

    nDiamantes = len(objetos)

    if (nDiamantes == 6):
        print "2 pesos"
    elif (nDiamantes == 5):
        print "5 pesos"
    elif (nDiamantes == 4):
        print "10 pesos"
    elif (nDiamantes == 3):
        print "20 pesos"
    elif (nDiamantes == 2):
        print "50 pesos"
    elif (nDiamantes == 1):
        print "100 pesos"


    

  
