import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')
import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
basePath = sys.path[0]

# def infoPelota(parteCanchaHSV):
#     pastoMask = pdi.segmentador(parteCanchaHSV,[35,100,50],[55,255,220])

def obtenerMitadCancha(imgHSV):
    "Obtiene la mitad de la cancha Xc"
    
    lineaMask= pdi.segmentador(imgHSV,[0,0,250], [10,10,255])
    lineaMask = cv2.morphologyEx(lineaMask,cv2.MORPH_ERODE,cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
    # plt.title("Linea")
    # plt.imshow(lineaMask,cmap='gray')
    # plt.show()

    _,lines = pdi.hough_Transform(lineaMask,100,-1,1)
    # print lines
    mitadCanchaEnX = 0
    cantidadLineas = len(lines)
    if (cantidadLineas > 1):
        promedio = 0
        for l in lines:
            promedio += l[0]
        mitadCanchaEnX = promedio / cantidadLineas
    else:
        mitadCanchaEnX = lines[0][0]

    # print "La mitad de la cancha esta en : " , mitadCanchaEnX
    return mitadCanchaEnX

def analizarJugadoraPelota(jugador,imgHSV):
    jugador_color = pdi.masking(imgHSV,jugador)
    esAzul = pdi.segmentador(jugador_color ,[110,215,188],[118,220,190])
    estaPelota = pdi.segmentador(jugador_color ,[28,250,250],[32,255,255])

    esEquipo = 0
    pelotaPertenece = 0


    if (pdi.infoSeg(esAzul,0.002)):
        esEquipo = 1

        if (pdi.infoSeg(estaPelota,0.000000001) ) :
            pelotaPertenece = 1
            print "PELOOOOTAAA ES DEL AZUL"

    else:

        esEquipo = 2
        if (pdi.infoSeg(estaPelota,0.00000001) ) :
            pelotaPertenece = 2
            print "PELOOOOTAAA ES DEL ROJOOOO"
    

    return esEquipo,pelotaPertenece



def analizarParteCancha(parteCancha):
    # canales = pdi.infoROI(parteCancha,False,True)
    pastoMask = pdi.segmentador(parteCancha,[35,100,50],[55,255,220])

    pastoMask = pdi.filtroMorfologico(pastoMask,9)

    _,jugadoresMask = cv2.threshold(pastoMask,200,255,cv2.THRESH_BINARY_INV)
    jugadores = pdi.gestionarObjetos(jugadoresMask)
    # plt.title("Jugadores de la parte de la cancha")
    # plt.imshow(jugadoresMask,cmap='gray')
    # plt.show()
    equipo1 = 0
    equipo2 = 0
    tienPelota = 0

    for jugador in jugadores:
        nroEquipo,tienePelotaJ = analizarJugadoraPelota(jugador.obtenerMascara(parteCancha),parteCancha)
        if nroEquipo == 1:
            equipo1 += 1            
        if nroEquipo == 2:
            equipo2 +=1
        if tienePelotaJ:
            tienPelota = tienePelotaJ


    # print "Cantidad de jugadores : ", len(jugadores)

    # plt.figure()
    # plt.imshow(jugadoresMask,cmap='gray')
    # plt.show()

    return equipo1,equipo2,tienPelota


if __name__ == "__main__":
    nameFile = "train2.png"
    img = cv2.imread(basePath+"/"+nameFile)

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgGRAY = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    Xc = int( obtenerMitadCancha(imgHSV))
    # Xc = 635
    
    
    parteIzquierda = imgHSV[:,:Xc-20]
    parteDerecha = imgHSV[:,Xc+20:]
    
    print "**************   INFO DEL PARTIDO   **********************"
    cantIzqJug1,cantIzqJug2,estaPelotaIzq = analizarParteCancha(parteIzquierda)
    cantDerJug1,cantDerJug2,estaPelotaDer = analizarParteCancha(parteDerecha)
    print   "---------------IZQUIERDA--------------------"
    print "Jugadores del Equipo Azul : ", cantIzqJug1
    print "Jugadores del Equipo Rojo : ", cantIzqJug2
    print   "--------------- DERECHA--------------------"
    print "Jugadores del Equipo Azul : ", cantDerJug1
    print "Jugadores del Equipo Rojo : ", cantDerJug2
    print   "-----------------------------------"

    # if estaPelotaIzq:
    #     print "La pelota esta en la parte izquierda y la tiene el equipo : " , estaPelotaIzq
    # else:
    #     print "La pelota esta en la parte derecha y la tiene el equipo : " , estaPelotaIzq







# INFO
# Azul :
# 	[110,215,188],[118,220,190]
	
# Pasto:
# 	[35,100,50],[55,255,220]

# Pelota:
# 	[28,250,250],[32,255,255]