import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
basePath = sys.path[0]



                

img = cv2.imread(basePath+"/1.png")
fichIMG = cv2.imread(basePath+"/fich.PNG")


imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
original = imgRGB.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# SEGMENTACION 
canales = pdi.infoROI(imgHSV,False,True)
mask_campo = pdi.autoSegmentador(imgHSV,canales,[0.02,0.00,0.001])

_ ,mask_line= cv2.threshold(imgGray,150,255,cv2.THRESH_BINARY)

# DETECCION DE BORDES - TRANSFORMADA DE HOUGH

verticalLines,linees = pdi.hough_Transform(np.uint8( mask_line),150,-10,10)
horizontalLines,linees2 = pdi.hough_Transform(np.uint8( mask_line),175,85,95)
Dx = linees2[0][0]
Dy = linees[0][0]

print "DistanciaV : ",Dy
print "DistanciaH : ",Dx

# SE PEGA LA PUBLICIDAD

_ , noCancha = cv2.threshold( mask_campo,150,255,cv2.THRESH_BINARY_INV)

M,N = fichIMG.shape[:2]
p1x = Dy - N  +1
p1y = Dx + 3
fichIMG = cv2.cvtColor(fichIMG,cv2.COLOR_BGR2RGB)

imagenConPublicidad = pdi.pasteImg(imgRGB,p1x,p1y,fichIMG)

resultado = pdi.pegar(imagenConPublicidad,noCancha,original)


# GRAFICOS

plt.subplot(121),plt.imshow(mask_campo,cmap='gray')
plt.subplot(122),plt.imshow(noCancha,cmap='gray')
plt.figure()
plt.subplot(121),plt.imshow(verticalLines,cmap='gray')
plt.subplot(122),plt.imshow(horizontalLines,cmap='gray')
plt.figure("Resultado")
plt.imshow(resultado)
plt.show()