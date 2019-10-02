import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pdifunFixed import *
import os, time

# Las funciones reciben banderas (numeros enteros)
# img2=cv.imread("img/camino.tif",cv.IMREAD_GRAYSCALE) # escala de grises




# -------------------------------------  EJERCICIO 1  -------------------------------------
# Ejercicio 1: Obtencion de informacion de formato y opciones de visualizacion.
#   1. Cargue y visualice diferentes imagenes.
#   2. Muestre en pantalla informacion sobre las imagenes.
#   3. Identifique y recorte una subimagen de una imagen (ROI, por Region Of Interest).
#   4. Investigue como mostrar en una sola ventana varias imagenes.
#

# # Read image
# im = cv2.imread('img/chairs.jpg')
#
# # Select ROI
# r = cv2.selectROI("Recartamela",im)
# # Format (x1 , y1 , dx1 , dy1)
#

# # Crop image
# imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
# imagen_a_mostar = np.append(imCrop,imCrop,axis=1)
# # Display cropped image
# plt.imshow(imagen_a_mostar,cmap='gray')
# plt.show()



# -------------------------------------  EJERCICIO 2  -------------------------------------
# Ejercicio 2: Informacion de intensidad.
# 1. Informe los valores de intensidad de puntos particulares de una imagen.
# 2. Obtenga y grafique un perfil de intensidad.
# 3. Grafique un perfil de intensidad a lo largo de un segmento de interes cualquiera.




# imagen = cv.imread('img/billete.jpg')
# y = get_intensity_line(imagen,5,0)
# print y
#
# plt.plot(range(len(y)),y)
# plt.ylim(0,256)
# plt.show()
#

# -------------------------------------  EJERCICIO 3  -------------------------------------
# Ejercicio 3: Efectos de la resolucion espacial
# 1. Cargue la imagen rmn.jpg
# 2. Modifique la resolucion espacial mediante submuestreo sucesivo con un factor
# de 2. Visualice el resultado con tamano normalizado al de la imagen original.

#
# img = cv.imread("img/rmn.jpg")
#
# plt.subplot(131), plt.imshow(img,cmap='gray'),plt.title("Original")
# plt.subplot(132), plt.imshow(sampling(img,2),cmap='gray'),plt.title("Muestreado x2 ")
# plt.subplot(133), plt.imshow(sampling(img,16),cmap='gray'),plt.title("Muestreado x16 ")
# plt.show()

# --------------------------  EJERCICIO 4 --------------------------




#
# img = cv.imread("img/huang2.jpg")
# levels = 20
# plt.subplot(121), plt.imshow(img,cmap='gray'), plt.title("Original")
# plt.subplot(122), plt.imshow(quantum(img,levels),cmap='gray'),plt.title("Cuantizada con " + str(levels))
# plt.show()


# --------------------------  EJERCICIO 5 - APLICACION --------------------------

# bottles = plt.imread("img/botellas.tif",0)
# plt.imshow(bottles,cmap='gray')
# plt.show()
# list_percent_empty = get_info_bottles(bottles,20)
#
# print "La botella mas vacia es : " , list_percent_empty.index(max(list_percent_empty))+1 , ". Llenado hasta un ",100-max(list_percent_empty)
