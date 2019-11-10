import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
basePath = sys.path[0]




if __name__ == '__main__':

    # nameImage = "/B2C1_01.jpg"
    # nameImage = "/B2C1_02a.jpg"
    # nameImage = "/B5C1_01.jpg"
    # nameImage = "/B5C1_02a.jpg"
    # nameImage = "/B10C1_01.jpg"
    # nameImage = "/B10C1_02a.jpg"
    # nameImage = "/B20C1_01a.jpg"
    nameImage = "/B20C1_02.jpg"
    # nameImage = "/B50C1_01a.jpg"
    # nameImage = "/B50C1_02.jpg"
    # nameImage = "/B100C1_01.jpg"
    # nameImage = "/B100C1_02a.jpg"

    img1 = cv2.imread(basePath+nameImage )
    img1 = pdi.equalizarIMG(img1)

    imgHSV = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)

    
    result2 = pdi.segmentador(imgHSV,[100,158,0],[117,245,255])
    result5 = pdi.segmentador(imgHSV,[50,50,0],[85,170,130])
    result10 = pdi.segmentador(imgHSV,[0,0,0],[180,210,73])
    result20 = pdi.segmentador(imgHSV,[175,165,20],[180,245,145])
    result50 = pdi.segmentador(imgHSV,[0,0,40],[107,74,186])
    result100 = pdi.segmentador(imgHSV,[100,84,14],[120,182,170])
    u = 0.1
    if infoSeg(result2,u):
        print "Son 2 pesos"
    if infoSeg(result5,u):
        print "Son 5 pesos"
    if infoSeg(result20,u):
        print "Son 10 pesos"

    if infoSeg(result50,u):
        print "Son 50 pesos"

    if infoSeg(result100,u):
        print "Son 100 pesos"

    

    


    # 2pe = [110,70,120]-[107,145,220]
    # 5pe = 11-55
    # 10pe = 0-25


    # img2 = cv2.imread(basePath+"/B2C1_02a.jpg" )
    # res = pdi.compararHistogramas(img1,img2) 
    # if res > 0.72 :
    #     print "Son distintos" , res*100,"%"," de diferencia"
    # else:
    #     print "Son parecidos", res*100,"%"," de diferencia"
    
    