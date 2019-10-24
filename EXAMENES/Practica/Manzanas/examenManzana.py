import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt


class imgObject:

    indx = -1
    Pcenter = None
    area = None
    detectorRect = (0,0,0,0)
    moments = None
    contourn = None


    def __init__(self,contour,indx):
        self.contourn = contour 
        self.indx = indx
    
    def obtenerArea(self):

        if (self.area == None):
            self.area = cv2.contourArea(self.contour)

        return self.area


        
    def obtenerRectDetector(self):

        if (self.detectorRect == None):
            self.detectorRect = cv2.boundingRect(self.contourn)

        return self.detectorRect

    def obtenerCentroObjeto(self):
        
        if (self.moments  != None):
            self.moments = cv2.moments(self.contour)
            m = self.moments
            self.Pcenter = ( int(m['m10'] /m['m00'] ) , int(m['m01'] /m['m00']) )
        
        return self.Pcenter
        






def gestionarObjetos(mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    listObjects = []

    for i in range(len(contours)):
        
        newObject = imgObject(contours[i],i)
        listObjects.append(newObject)

    return listObjects




basePath = sys.path[0]

img = cv2.imread(basePath+'/EXAMEN09.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# pdi.infoROI(img)

manzanas_rojas_mask = pdi.segmentador(img,[173,125,127],[180,240,211])


manzanas = gestionarObjetos(manzanas_rojas_mask)

print len(manzanas)


plt.imshow(manzanas_rojas_mask,cmap='gray'),plt.show()

# manzanas_verdes = pdi.segmentador(img,[40,172,127],[44,244,170])
# plt.imshow(manzanas_verdes,cmap='gray'),plt.show()





