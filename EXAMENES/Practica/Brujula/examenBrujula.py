import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
basePath = sys.path[0]

def filtroMorfologico(mask,k,iter=1):
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
    # mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,se,iterations=1)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,se,iterations=iter)
    return mask

def gestionarAngulo(mask):
    M,N = mask.shape[:2]
    centro = (M//2,N//2)
    puntosMask = pdi.gestionarObjetos(mask)
    if len(puntosMask) != 2:
        print "No se segmento correctamente"
        return

    puntos = []
    distancias = []
    for p in puntosMask:
        oCentro = p.obtenerCentroObjeto()
        # mask = p.dibujate(mask,"xasdfasdf")
        puntos.append(oCentro)
        distancias.append( pdi.dist(oCentro,centro)) 

    indMax = np.argmax( distancias)
    indMin = np.argmin( distancias)
    a = np.array([puntos[indMax][0] - centro[0],centro[1] -puntos[indMax][1]   ])
    b = np.array([puntos[indMin][0] - centro[0], centro[1] -puntos[indMin][1]   ])
    num =  np.dot(a,b)
    den = np.linalg.norm(a)* np.linalg.norm(b)
    
    angulo = math.acos(float(num/den)) * 180 / math.pi
    return 360-angulo

if __name__ == "__main__":
        
    # CARGA DE IMAGENES

    imgOriginal = cv2.imread(basePath+"/4.png")

    imgRGB = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2HSV)
    imgGRAY = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY)
    # pdi.infoROI(imgHSV)
    # plt.show()
    # print pdi.infoROI(imgHSV)
    mask_rojo = pdi.segmentador(imgHSV,[176,0,0],[182,255,255])
    _, bandasNegras = cv2.threshold(imgGRAY,50,255,cv2.THRESH_BINARY_INV)

    bandasNegras = filtroMorfologico(bandasNegras,17)
    mask_rojo = filtroMorfologico(mask_rojo,23,iter=2)

    puntos = cv2.bitwise_and(bandasNegras,mask_rojo)
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
    puntos = cv2.morphologyEx(puntos,cv2.MORPH_DILATE,se)
    plt.imshow(puntos,cmap='gray')
    plt.show()
    print "Angulo ", gestionarAngulo(puntos)


    # listObj = pdi.gestionarObjetos(mask_rojo)

    # M,N = imgOriginal.shape[:2]
    # C = (int(0.5*M),int(0.5*N))

    # centros = []
    # for o in listObj:
    #     centrO =  o.obtenerCentroObjeto()
    #     if (centrO != None):
    #         centros.append(centrO)


    # # mask_rojo =  cv2.morphologyEx(mask_rojo, cv2.MORPH_DILATE, (200,200,     ), iterations=50)

    # print "Centros " , centros
    # plt.imshow(mask_rojo,cmap='gray')
    # plt.show()

