import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')
import math
import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
basePath = sys.path[0]


def obtenerCentrosAsteroide(imgHSV):
    
    asteroideMask = pdi.segmentador(imgHSV,[0,190,150],[0,255,250])
    asteroideMask = cv2.morphologyEx(asteroideMask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    asteroide = pdi.gestionarObjetos(asteroideMask)
    
    centroAsteroide = pdi.gestionarObjetoMaximaArea(asteroide)
    print "Centro del asteroide: ", centroAsteroide.obtenerCentroObjeto()
    return centroAsteroide.obtenerCentroObjeto()

def calcularAngulo(P1,C,P2):
    """
    Obtiene el angulo entre 3 puntos que forman dos vectores que comparten C en comun
    return: Grados entre dos vectores
    """

    a = np.array([P1[0] - C[0], P1[1]- C[1]   ])
    b = np.array([P2[0] - C[0], P2[1]- C[1]   ])

    num =  np.dot(a,b)
    den = np.linalg.norm(a)* np.linalg.norm(b)
    
    angulo = math.acos(float(num/den)) * 180 / math.pi
    return angulo    

if __name__ == "__main__":

    nameFile = "Avion5.png"
    
    img = cv2.imread(basePath+"/"+nameFile)

    imgGRAY = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)



    #Se obtiene el centro del asteroide
    centroAsteroide  = obtenerCentrosAsteroide(imgHSV)
    # --------------------------------------
    # Luego se procede a obtener el angulo de la trayectoria
    imgGRAY = pdi.equalizarIMG(imgGRAY)

    # _,trayectoMask = cv2.threshold(imgGRAY,175,255,cv2.THRESH_BINARY)
    trayectoMask = pdi.segmentador(imgHSV,[0,0,200],[255,255,255])

    trayectoMask = cv2.morphologyEx(trayectoMask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
    trayectoMask = cv2.morphologyEx(trayectoMask,cv2.MORPH_ERODE,cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)),iterations=4)
    imgLine,lines = pdi.hough_Transform(trayectoMask,50)


    plt.imshow(trayectoMask,cmap='gray')
    # plt.imshow(imgRGB)
    plt.show()

    if len(lines) > 10:
        plt.imshow(imgLine)
        plt.show()
        exit("Demasiadas lineas")
        
    elif (len(lines) == 0):
        exit("No se encontraron lineas")

    print lines
    print calcularAngulo(lines[0][0],lines[0][1],centroAsteroide)
    # print calcularAngulo([0,100],[500,100],[500,0])

    # plt.imshow(imgGRAY,cmap='gray')
    # # plt.imshow(imgRGB)
    # plt.show()