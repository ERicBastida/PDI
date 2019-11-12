import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
basePath = sys.path[0]


def maximaArea(objetos):
    objeto = None
    areas = []
    for o in objetos:
        areas.append(o.obtenerArea())
    i = np.argmax(areas)
    objeto = objetos[i]
    return objeto

if __name__ == "__main__":
    nameFile = "/34.jpg"
    img = cv2.imread(basePath+nameFile)

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgGRAY = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # pdi.infoROI(imgHSV)

    cervezaLagerMask = pdi.segmentador(imgHSV,[18,0,128],[26,240,240]) 
    cervezaStoutMask = pdi.segmentador(imgHSV,[0,0,0],[35,150,150]) 
    _, birraMask = cv2.threshold(imgGRAY,220,255,cv2.THRESH_BINARY_INV)
    # #Mascara de la birra
    birraMask = pdi.filtroMorfologico(birraMask,9)
    cervezaStoutMask = pdi.filtroMorfologico(cervezaStoutMask,15,iter=2)
    cervezaLagerMask = pdi.filtroMorfologico(cervezaLagerMask,15,iter=2)
    # birraMask = pdi.filtroMorfologico(birraMask,9)
    birras = pdi.gestionarObjetos(birraMask)

    birra = maximaArea(birras)
    birraLagers = pdi.gestionarObjetos(cervezaLagerMask)
    birraStouts = pdi.gestionarObjetos(cervezaStoutMask)

    birraLager = maximaArea(birraLagers)
    birraStout = maximaArea(birraStouts)

    areaTotal = birra.obtenerArea()
    areaLager = birraLager.obtenerArea() / areaTotal
    areaStout = birraStout.obtenerArea() / areaTotal
    print areaTotal
    print areaStout
    print areaLager
    areaEspuma = 0
    areaCerveza= 0
    if (areaLager > 0.4):
        print "Es Lager"
        areaEspuma = 1- areaLager
        areaCerveza = areaLager
        
    else:
        print "Es Stout"
        areaEspuma = 1- areaStout
        areaCerveza = areaStout

    print "Porcentaje de Cerveza: ", areaCerveza
    print "Porcentaje de Espuma: ", areaEspuma

    plt.subplot(131)
    plt.imshow(cervezaLagerMask,cmap='gray')
    plt.subplot(132)
    plt.imshow(cervezaStoutMask,cmap='gray')
    plt.subplot(133)
    plt.imshow(birraMask,cmap='gray')
    # plt.imshow(imgRGB)



    # objecBirra = pdi.gestionarObjetos(maskBirra)[0]

    # print objecBirra.obtenerCentroObjeto()



    # plt.imshow(maskBirra,cmap='gray')
    plt.show()