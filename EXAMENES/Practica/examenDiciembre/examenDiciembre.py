import sys #La libreria de funciones esta 4 carpetas arriba
sys.path.insert(1, sys.path[0]+'../../../../')

import pdifunFixed as pdi
import cv2
from matplotlib import pyplot as plt
import numpy as np
import math
basePath = sys.path[0]

if __name__ == "__main__":
    nameFile = ""
    img = cv2.imread(basePath+"/"+nameFile)

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imgGRAY = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    pdi.infoROI(imgHSV)
    