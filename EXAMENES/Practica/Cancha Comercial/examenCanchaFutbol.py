import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
basePath = sys.path[0]

def autoSegmentador(img,infoCanales,umbrales):
   
    M,N = img.shape[:2]
    total = M*N

    U1 = umbrales[0]*total
    U2 = umbrales[1]*total
    U3 = umbrales[2]*total

    umbralCanalesMinimos = []
    umbralCanalesMaximos = []

    tC1  = []
    tC2  = []
    tC3  = []

    nCanales = len(infoCanales)
    nIntensities = len(infoCanales[0])
    
    for i in range(nIntensities):
        if infoCanales[0][i][0] > U1:
            tC1.append(i)
        if infoCanales[1][i][0] > U2:
            tC2.append(i)
        if infoCanales[2][i][0] > U3:
            tC3.append(i)

    umbralCanalesMinimos= [tC1[0],tC2[0],tC3[0]]
    umbralCanalesMaximos= [tC1[-1],tC2[-1],tC3[-1]]

    mask = pdi.segmentador(img,umbralCanalesMinimos,umbralCanalesMaximos)

    return mask

                

img = cv2.imread(basePath+"/1.png")
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
original = imgRGB.copy()
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

canales = pdi.infoROI(imgHSV,False,True)

mask_campo = autoSegmentador(imgHSV,canales,[0.02,0.00,0.001])
# mask_campo = cv2.morphologyEx(mask_campo,cv2.MORPH_CLOSE,(3,3),iterations=2)


_ ,mask_line= cv2.threshold(imgGray,150,255,cv2.THRESH_BINARY)

verticalLines,linees = pdi.hough_Transform(np.uint8( mask_line),150,-10,10)
horizontalLines,linees2 = pdi.hough_Transform(np.uint8( mask_line),175,85,95)
Dx = linees2[0][0]
Dy = linees[0][0]
print "DistanciaV : ",Dy
print "DistanciaH : ",Dx

_ , noCancha = cv2.threshold( mask_campo,150,255,cv2.THRESH_BINARY_INV)


fichIMG = cv2.imread(basePath+"/fich.PNG")
M,N = fichIMG.shape[:2]
p1x = Dy - N  +1
p1y = Dx + 3
fichIMG = cv2.cvtColor(fichIMG,cv2.COLOR_BGR2RGB)

imagenConPublicidad = pdi.pasteImg(imgRGB,p1x,p1y,fichIMG)

resultado = pdi.pegar(imagenConPublicidad,noCancha,original)

plt.subplot(121)
plt.imshow(mask_campo,cmap='gray')
plt.subplot(122)
plt.imshow(noCancha,cmap='gray')
plt.figure()
plt.subplot(121)
plt.imshow(verticalLines,cmap='gray')
plt.subplot(122)
plt.imshow(horizontalLines,cmap='gray')

plt.figure("Resultado")
plt.imshow(resultado)
plt.show()