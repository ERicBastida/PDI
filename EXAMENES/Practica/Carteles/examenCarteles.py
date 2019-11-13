import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
basePath = sys.path[0]


def contarCalles(mask):
    "Obtiene el inicio y fin de las flechas separadas horizontalmente y devuelve la cantidad de calles (arriba y abajo)"
    objectsCalles = pdi.gestionarObjetos(mask)
    centros = [flecha.obtenerCentroObjeto()[1] for flecha in objectsCalles]
    # print "Cantidad de calles: ", len(objectsCalles)
    print "Centros: ", centros
    arriba = min(centros)
    cArriba = 0
    abajo =max(centros)
    cAbajo = 0
    umbral = (arriba + abajo)//2
    print "Umbral ",umbral
    for c in centros:
        print "Centro: ",c
        if c > umbral:
            print "Uno abajo"
            cAbajo += 1
        else:
            cArriba +=1
            print "Uno arriba"
            

    return cArriba, cAbajo

def obtenerDireccion(mask):

    # plt.show()
    _,Uderecha = pdi.hough_Transform(mask,15,35,55)
    _,Uizquierda = pdi.hough_Transform(mask,15,35+90,55+90)
    # print "Derecha", len(Uderecha)
    # print "Izquierda", len(Uizquierda)
    
    if len(Uderecha)>len(Uizquierda):
        return True
    else:
        return False

if __name__ == "__main__":

    img = cv2.imread(basePath+"/2.jpg")
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # pdi.infoROI(imgHSV)
    segCartelMask = pdi.segmentador(imgHSV,[18,170,170],[25,250,215])
    segCartelMaskObj = pdi.gestionarObjetos(segCartelMask)
    cartel = pdi.gestionarObjetoMaximaArea(segCartelMaskObj)
    (Xo,Yo, ancho,alto) =  cartel.obtenerRectDetector()
    
    # Obtenemos la mascara que va a separar las flechas de arriba y las de abajo
    bandaMask = np.uint8(np.zeros(segCartelMask.shape[:2]))
    bandaMask[Yo+alto//4:Yo+alto//4+alto//2,Xo:Xo+ancho] = 1
    base = np.zeros(img.shape[:2])
    # Obtenemos el cartel unicamente
    maskCartel = np.uint8(cartel.obtenerMascara(base))
    cartelMask = cv2.bitwise_and(segCartelMask,maskCartel)
    # Invertimos los valores para realizar un or y dejar las puntas de arriba y abajo
    cartelMask = np.uint8(np.where(cartelMask > 1,1,0))
    resultado = cv2.bitwise_or(cartelMask,bandaMask)
    # Invertimos para contar los objetos
    calles = np.where(resultado > 0, 0,1)
    flechas = cv2.bitwise_and(np.uint8(calles),maskCartel)
    # Y por ultimo gestionamos dichos objetos para contar cuantos hay arriba y abajo
    arriba,abajo   = contarCalles(flechas)

    print "Calles Arriba: ", arriba
    print "Calles abajo: ", abajo
    # Se obtiene la parte central
    desvio = cv2.bitwise_and(cartelMask,bandaMask)
    desvio = np.where(desvio > 0, 0,1)
    desvio = cv2.bitwise_and(np.uint8(bandaMask),np.uint8(desvio))
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    desvio = cv2.morphologyEx(desvio,cv2.MORPH_ERODE,se)
    derecha = obtenerDireccion(desvio)
    if derecha:
        print "Doble a la derecha"
    else:
        print "Doble a la izquierda"

    plt.imshow(flechas,cmap='gray')
    plt.show()
