import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')
import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
basePath = sys.path[0]

def obtenerPuntosPorDado(dado,imgHSV):

    dadoHSV = pdi.masking(imgHSV,dado.obtenerMascara(imgHSV))

    puntosMask = pdi.segmentador(dadoHSV,[0,0,175],[200,50,255])
    puntosMask = cv2.morphologyEx(puntosMask,cv2.MORPH_ERODE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))

    puntos = pdi.gestionarObjetos(puntosMask)

    return len(puntos)




def gestionarDados(dados,imgHSV):

    if len(dados) == 0:
        exit("NO se detectaron dados")
    puntosDado = []

    for dado in dados:
        puntosDado.append(obtenerPuntosPorDado(dado,imgHSV))
    
    print 'Puntos por dado ', puntosDado
    print 'Cantidad de puntos obtendos : ', sum(puntosDado)



def obtenerMascaraDados(imgHSV):

    mesaMask = pdi.segmentador(imgHSV,[80,0,0],[92,255,255])
    _,mesaMask = cv2.threshold(mesaMask,200,255,cv2.THRESH_BINARY_INV)
    mesaMask = pdi.filtroMorfologico(mesaMask,5)
    # plt.imshow(mesaMask,cmap='gray')
    # plt.show()
    return mesaMask


if __name__ == "__main__":

    nameFile = "/dados4.jpg"
    
    img = cv2.imread(basePath+nameFile)

    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgGRAY = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    dadosMask = obtenerMascaraDados(imgHSV)
    dados= pdi.gestionarObjetos(dadosMask)

    gestionarDados(dados,imgHSV)





