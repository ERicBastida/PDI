import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
basePath = sys.path[0]




if __name__ == "__main__":

    img = cv2.imread(basePath+"/3.jpg")
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # pdi.infoROI(imgHSV)
    cartelMask = pdi.segmentador(imgHSV,[18,170,170],[25,250,215])
    cartelMasks = pdi.gestionarObjetos(cartelMask)
    cartelMask = pdi.gestionarObjetoMaximaArea(cartelMasks)

    base = np.zeros(img.shape[:2])
    base = cartelMask.obtenerMascara(base)

    plt.imshow(base,cmap='gray')
    plt.show()
