import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
basePath = sys.path[0]



img = cv2.imread(basePath+'/EXAMEN09.jpg')
imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# pdi.infoROI(img)

manzanas_rojas_mask = pdi.segmentador(imgHSV,[173,125,127],[180,240,211])
manzanas_verdes_mask = pdi.segmentador(imgHSV,[39,150,124],[44,237,172])


manzanas_rojas = pdi.gestionarObjetos(manzanas_rojas_mask)
manzanas_verdes = pdi.gestionarObjetos(manzanas_verdes_mask)

print len(manzanas_rojas), " manzanas rojas."
print len(manzanas_verdes), " manzanas verdes."

areaVerdes = []
areaRojas = []

for indx in range(len(manzanas_verdes)):
    areaVerdes.append(manzanas_verdes[indx].obtenerArea())
    cv2.putText(imgRGB,"V: "+ str(indx), manzanas_verdes[indx].obtenerCentroObjeto(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

for indx in range(len(manzanas_rojas)):
    areaRojas.append(manzanas_rojas[indx].obtenerArea())
    cv2.putText(imgRGB,"R: "+str(indx), manzanas_rojas[indx].obtenerCentroObjeto(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

indxRoja = areaRojas.index(min(areaRojas))
indxVerde = areaVerdes.index(min(areaVerdes))
imgRGB = manzanas_verdes[indxVerde].dibujate(imgRGB)
imgRGB = manzanas_rojas[indxRoja].dibujate(imgRGB)

print "La manzana roja numero" , indxRoja, " es la mas chica"
print "La manzana verde numero" , indxVerde, " es la mas chica"



plt.imshow(imgRGB)

plt.show()







