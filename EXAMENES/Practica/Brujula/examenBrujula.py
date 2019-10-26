import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
basePath = sys.path[0]


# CARGA DE IMAGENES

imgOriginal = cv2.imread(basePath+"/1.png")

imgRGB = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2RGB)
imgHSV = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2HSV)

# print pdi.infoROI(imgHSV)
mask_rojo = pdi.segmentador(imgHSV,[0,150,130],[15,255,200])
listObj = pdi.gestionarObjetos(mask_rojo)

M,N = imgOriginal.shape[:2]
C = (int(0.5*M),int(0.5*N))

centros = []
for o in listObj:
    centrO =  o.obtenerCentroObjeto()
    if (centrO != None):
        centros.append(centrO)


# mask_rojo =  cv2.morphologyEx(mask_rojo, cv2.MORPH_DILATE, (200,200,     ), iterations=50)

print "Centros " , centros
plt.imshow(mask_rojo,cmap='gray')
plt.show()

