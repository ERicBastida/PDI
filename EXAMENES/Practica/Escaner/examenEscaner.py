import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')
import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
basePath = sys.path[0]

def anguloPromedio(lines):
    "Obtiene la lista de rectas en coordenadas polares"
    cantLines = len(lines)
    if ( cantLines == 0):
        exit("No se puede calcular el angulo promedio sin rectas")
    anguloPromedio = 0

    for line in lines:
        anguloPromedio += line[1]
    return (anguloPromedio/cantLines) * 180 /math.pi

if __name__ == "__main__":
    nameFile = "/escaneo3.jpg"
    # nameFile = "/escaneo1.jpg"

    img = cv2.imread(basePath+nameFile)

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgGRAY = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    imgGRAY = cv2.blur(imgGRAY,(3,3))

    bordes = cv2.Canny(imgGRAY, 175, 225, apertureSize=5)
    thresh = int(250)
    imgLines,lines1 = pdi.hough_Transform(bordes,thresh,-10,10)
    imgLines,lines2 = pdi.hough_Transform(bordes,thresh,-10+180,10+180)
    print "Angulo promedio : ", anguloPromedio(lines1)
    print "Angulo promedio : ", anguloPromedio(lines2)
    # plt.subplot(121)
    # print len(lines)
    imgGRAY = pdi.rotar(imgGRAY,anguloPromedio(lines1)-2)
    plt.imshow(imgGRAY,cmap='gray')
    # plt.subplot(122)
    # plt.imshow(bordes,cmap='gray')
    plt.show()