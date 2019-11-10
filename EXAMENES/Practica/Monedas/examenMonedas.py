import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
basePath = sys.path[0]

def gestionarMonedasLaton(monedasLaton,imgRGB):
    suma = 0
    for moneda in monedasLaton:
        area = moneda.obtenerArea()
       
        if area > 6500:
            imgRGB = moneda.dibujate(imgRGB,"50c")
            suma += 0.5
        elif area > 4500:
            imgRGB = moneda.dibujate(imgRGB,"20c")
            suma += 0.2
        elif area >500 and area < 4500:
            imgRGB = moneda.dibujate(imgRGB,"10c")
            suma += 0.1

    return imgRGB, suma

def gestionarMonedasCobreAcero(monedasCobre,imgRGB):
    suma = 0
    for moneda in monedasCobre:
        area = moneda.obtenerArea()
        if area > 4500:
            imgRGB = moneda.dibujate(imgRGB,"5c")
            suma += 0.05
        elif area > 3200:
            imgRGB = moneda.dibujate(imgRGB,"2c")
            suma += 0.02
        elif area >500 and area < 3200:
            imgRGB = moneda.dibujate(imgRGB,"1c")
            suma += 0.01

    return imgRGB, suma

def gestionarMonedas(mask,imgRGB):

    suma = 0    
    monto = 0.0
    monedas = pdi.gestionarObjetos(mask)

    monedasCobre = []
    monedasLaton = []
    monedasCuproLaton = []

    for moneda in monedas:

        monedaMask = moneda.obtenerMascara(imgRGB)
        monedaIMG = pdi.masking(imgRGB,monedaMask)
        monedaHSV = cv2.cvtColor(monedaIMG,cv2.COLOR_RGB2HSV)

        areaMoneda = moneda.obtenerArea()
        
        cobreAcero  = pdi.segmentador(monedaHSV,[0,58,80],[8,147,202])
        areaCobreAcero = (cobreAcero > 0 ).sum()
        laton       = pdi.segmentador(monedaHSV,[11,40,43],[23,115,245])
        areaLaton = (laton > 0 ).sum()
        cuproniquel = pdi.segmentador(monedaHSV,[120,5,56],[164,33,220])
        areaCuproniquel = (cuproniquel > 0 ).sum()

        areaCobreAcero = round(areaCobreAcero/areaMoneda,2)
        areaLaton = round(areaLaton/areaMoneda,2)
        areaCuproniquel = round(areaCuproniquel/areaMoneda,2)

        # print suma,": ", areaCobreAcero,areaLaton,areaCuproniquel
        if (areaCobreAcero > 0.7):
            monedasCobre.append(moneda)

        elif(areaLaton > 0.7):
            monedasLaton.append(moneda)
    
        else:
            if(areaCuproniquel > areaLaton):
                imgRGB = moneda.dibujate(imgRGB,"1pe ")
                monto += 1.0

            else:
                imgRGB = moneda.dibujate(imgRGB,"2pe ")
                monto += 2.0

            monedasCuproLaton.append(moneda)
            
            
        suma +=1



    imgRGB, montoCA = gestionarMonedasCobreAcero(monedasCobre,imgRGB)
    imgRGB, montoL = gestionarMonedasLaton(monedasLaton,imgRGB)
    print montoCA
    print montoL
    print monto
    suma = monto + montoL + montoCA
    return imgRGB,suma


        
if __name__ == "__main__":
        
    nameFile = "/03_Monedas.jpg"
    img = cv2.imread(basePath+nameFile)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    #-----------------------------Obtencion de info-----------------------------
    # pdi.infoROI(imgHSV)
    # plt.show()
    # Acero con Cobre : 1 2 5
    # [0,58,80],[8,147,202]
    # Laton: 10, 20, 50 
    # [11,40,43],[23,115,245]
    # Cuproniquel
    # [120,5,56],[164,33,220]

    #---------------------------------------------------------------------------
    _,maskMonedas = cv2.threshold(imgGray,200,255,cv2.THRESH_BINARY_INV)
    maskMonedas = cv2.morphologyEx(maskMonedas,cv2.MORPH_CLOSE,(5,5))
    imgRBG,monto = gestionarMonedas(maskMonedas,imgRGB)
    plt.imshow(imgRBG)
    print "Total: ", monto
    plt.show()


    
 