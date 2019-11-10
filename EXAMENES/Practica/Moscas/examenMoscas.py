import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
basePath = sys.path[0]




def mascaraMoscas(img):
    "Obtiene una binaria de moscas y devuelve la cantudad "

    _,moscasMask = cv2.threshold(img,50,255,cv2.THRESH_BINARY_INV)

    element = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    #Para sacar las patas y mugre al rededor
    moscasMaskClose = cv2.morphologyEx(moscasMask,cv2.MORPH_OPEN,element)
    #Para que esten conectados
    moscasMaskClose = cv2.morphologyEx(moscasMaskClose,cv2.MORPH_DILATE,element)

    # moscas = pdi.gestionarObjetos(moscasMaskClose)

    return moscasMaskClose



if __name__ == "__main__":
 
    nameFile = "/Platos04.jpg"
    img = cv2.imread(basePath+nameFile)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #-----------------------Info---------------------------
    # pdi.infoROI(imgHSV)
    # plt.show()
    # Mesa
    # [177,205,202], [180,210,210]
    # Zapallito
    # [10,249,215], [14,255,225]
    # De la casa
    # [17,110,126], [28,213,200]

    # ---------------------Init-----------------------------
    # Se obtiene el plato
    _,plato = cv2.threshold(imgGray,200,255,cv2.THRESH_BINARY)
    # Se obtiene la mesa de color rojo
    mesa = pdi.segmentador(imgHSV,[177,200,0], [180,210,255])    
    # Se obtiene lo contrario a la mesa (plato y moscas)
    _,mesa = cv2.threshold(mesa,200,255,cv2.THRESH_BINARY_INV)
    # Se procede a obtener todo el plato
    seed = np.uint8(np.zeros(mesa.shape[:2]))
    seed[500,500]= 1
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    # Tengo la mascara del plato, mesa y moscas
    platoMask = pdi.morphologicalReconstructionbyDilation(seed,mesa,se,100)
    platoTemp = platoMask*255
    _,mesaMask = cv2.threshold(platoTemp,100,255,cv2.THRESH_BINARY_INV)
    moscasMask = np.uint8(mascaraMoscas(imgGray))

    sopaZapallo = pdi.segmentador(imgHSV,[10,249,215], [14,255,225])
    sopaCasa = pdi.segmentador(imgHSV,[17,110,126], [28,213,200])
    sopa = None
    sopaCasaBandera = False
    if ((sopaZapallo > 100).sum() > 100):
        print "Es sopa de zapallo"
        sopa = sopaZapallo
    else:
        print "Es sopa de la Casa"
        sopa = sopaCasa
        sopaCasaBandera = True
        
    
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    #Para sacar las patas y mugre al rededor
    sopaMask = np.uint8(cv2.morphologyEx(sopa,cv2.MORPH_CLOSE,element,iterations=3))

    moscasTotales = len(pdi.gestionarObjetos(moscasMask))
    print "Moscas totales" , moscasTotales
    moscasEnMesa = cv2.bitwise_and(mesaMask,moscasMask)
    moscasEnSopa = cv2.bitwise_and(sopaMask,moscasMask)
    cantMoscasEnMesa = len(pdi.gestionarObjetos(moscasEnMesa))
    cantMoscasEnSopa = len(pdi.gestionarObjetos(moscasEnSopa))
    cantMoscasPlato = moscasTotales - cantMoscasEnMesa - cantMoscasEnSopa
    print "Moscas en Mesa" , cantMoscasEnMesa
    print "Moscas en Sopa" , cantMoscasEnSopa
    print "Moscas en Plato", cantMoscasPlato

    if sopaCasaBandera:
        if cantMoscasPlato < 5:
            print "Esta bien servido"
        else:
            print "Esta mal servido"
    else:
        if cantMoscasPlato < 4:
            print "Esta bien servido"
        else:
            print "Esta mal servido"
