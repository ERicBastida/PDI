#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

from __future__ import division
import cv2 
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import ndimage

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
            self.area = cv2.contourArea(self.contourn)

        return self.area


        
    def obtenerRectDetector(self):

        if (self.detectorRect == None):
            self.detectorRect = cv2.boundingRect(self.contourn)

        return self.detectorRect

    def obtenerCentroObjeto(self):
        
        if (self.moments  == None):
            self.moments = cv2.moments(self.contourn)
            m = self.moments
            if m['m00'] == 0:
                print "No se pudo obtener el centro de un objeto"    
            else:
                self.Pcenter = ( int(m['m10'] /m['m00'] ) , int(m['m01'] /m['m00']) )
        
        return self.Pcenter

    def dibujate(self,img):
        cv2.drawContours(img, [self.contourn], -1, (0,255,255), 1, 8)
        return img
        


def gestionarObjetos(mask):
    "Recibe una mascara binaria y envia una lista de los objetos detectados"
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    listObjects = []

    for i in range(len(contours)):
        
        newObject = imgObject(contours[i],i)
        listObjects.append(newObject)

    return listObjects

def pasteImg(baseImage,x_offset , y_offset, image):
    "Pega una imagen (image) sobre una imagen base (baseImage) segun el offset que se ingrese"
    x_offset = int(x_offset)
    y_offset = int(y_offset)
    baseImage[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image

    return baseImage

def infoROI(img,show=True,all=False):
    
    image = img.copy()
    
    if all == False:
        # Select ROI
        r = cv2.selectROI(img)
        # Crop image
        image = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    M,N = image.shape[:2]
    print "El tamano de lo recortado es ", M, " x ", N
    C = 1
    if (len(img.shape) == 3):
        _,_,C = img.shape


    _, axs = plt.subplots(int(C),1)
    histograms = []
    if C == 1:
        histC = histograma(image)
        axs.stem(range(len(histC)),histC,markerfmt=" ")
        axs.set_title("Channel "+ str(0))
        histograms = histC
    else:
        for c in range(C):

            histC = histograma(image[:,:,c])
                
            axs[int(c)].stem(range(len(histC)),histC,markerfmt=" ")
            axs[int(c)].set_title("Channel "+ str(c))
            histograms.append(histC)


    return histograms 

    
def segmentador(img,lowerColor,upperColor):
    """
    img: La imagen debe estar formateada en el MODELO en especifico
    lowerColor: Valor mínimo en el MODELO [Min1,Min2,Min3]
    upperColor: Valor máximo en el MODELO [Max1,Max2,Max3]
        
    """
    try:
        # define range of red color in HSV
        lower_color = np.array(lowerColor )
        upper_color = np.array(upperColor)

        # Threshold the HSV image to get only blue colors
        G = cv2.inRange(img, lower_color, upper_color)

    except e:
        print "Ha ocurrido un error en el segmentador: \n" 
        return e

    return G


def histograma(img):
    "Devuelve un vector con la cantidad de pixeles por intensidad de color"
    result = cv2.calcHist([img],[0],None,[256],[0,256])
    return result

def histograma3C(img):
    hisA = cv2.calcHist([img],[0],None,[256],[0,256])
    hisB = cv2.calcHist([img],[1],None,[256],[0,256])
    hisC = cv2.calcHist([img],[2],None,[256],[0,256])
    
    plt.plot(range(len(hisA)), hisA,'r')
    plt.plot(range(len(hisB)), hisB,'g')
    plt.plot(range(len(hisC)), hisC,'b')

    return hisA,hisB,hisC

def normalizar(img):
    M,N = img.shape[:2]
    for i in range(M):
        for j in range(N):
            if (img[i,j]>255):
                img[i,j] = 255
            elif(img[i,j]<0):
                img[i,j] = 255
    return img

def sumaIMG(imgs):
    N = len(imgs)
    acumIMG = np.zeros(imgs[0].shape)
    for img in imgs:
        acumIMG += img 
    acumIMG =  acumIMG/N
    return normalizar( acumIMG)

def sampling(img,factor):
    "Muestrea la imagen seg�n el factor ingresado"
    M,N,_ = img.shape

    step_x = M//factor
    step_y = N//factor
    IMG = np.zeros((step_x,step_y))

    for i in range(step_x):
        for j in range(step_y):
            # Ver el tema de interpolaciones linear, bilineal  y bicubica
            IMG[i,j] = np.average(img[i*factor:i*factor + factor,j*factor:j*factor + factor ])

    return IMG

def optimalDFTImg(img):
    """Zero-padding sobre img para alcanzar un tama�o �ptimo para FFT"""
    h = cv2.getOptimalDFTSize(img.shape[0])
    w = cv2.getOptimalDFTSize(img.shape[1])
    return cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, w - img.shape[1], cv2.BORDER_CONSTANT)

def spectrum(img):
    """Calcula y muestra el modulo logartimico de la DFT de img."""
    # img=optimalDFTImg(img)

    imgf = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    modulo = np.log(cv2.magnitude(imgf[:, :, 0], imgf[:, :, 1]) + 1)
    modulo = np.fft.fftshift(modulo)
    modulo = cv2.normalize(modulo, modulo, 0, 1, cv2.NORM_MINMAX)

    return modulo

def rotar(img,angle):
    return ndimage.rotate(img,angle)

def rotate(img, angle):
    """Rotacion de la imagen sobre el centro"""
    if (angle != 0):
        r = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle, 1.0)
        result = cv2.warpAffine(img, r, img.shape)
        return result
    else:
        return img


def dist(a, b):
    """distancia Euclidea"""
    return np.linalg.norm(np.array(a) - np.array(b))

def filterGaussian(rows, cols, corte, PA = False):
    """Filtro de magnitud gausiano"""

    magnitud = np.zeros((rows, cols))

    corte *= rows
    for k in range(rows):
        for l in range(cols):
            if PA:
                magnitud[k, l] = 1-np.exp(-dist([k, l], [rows // 2, cols // 2]) / 2 / corte / corte)
            else:

                magnitud[k, l] = np.exp(-dist([k, l], [rows // 2, cols // 2]) / 2 / corte / corte)

    return np.fft.ifftshift(magnitud)

def filterIdeal(rows, cols, corte):
    """filtro de magnitud ideal"""
    magnitud = np.zeros((rows, cols))
    magnitud = cv2.circle(magnitud, (cols // 2, rows // 2), int(rows * corte), 1, -1)
    # np.fft.ifft2
    return np.fft.ifftshift(magnitud)

def filterButterworth(rows, cols, corte, order):
    """filtro de magnitud Butterworth"""
    # corte = w en imagen de lado 1
    # 1 \over 1 + {D \over w}^{2n}
    magnitud = np.zeros((rows, cols))
    corte *= rows
    for k in range(rows):
        for l in range(cols):
            d2 = dist([k, l], [rows // 2, cols // 2])
            magnitud[k, l] = 1.0 / (1 + (d2 / corte / corte) ** order)

    return np.fft.ifftshift(magnitud)

def motionBlur(size, a, b):
    """Filtro de movimiento en direcciones a y b"""
    transformation = np.zeros(size)
    rows = size[0]
    cols = size[1]

    # fase exp(j\pi (ua + vb))
    # magnitud \frac{ \sin(\pi(ua+vb)) }{ \pi (ua+vb) }
    for k in range(rows):
        for l in range(cols):
            u = (l - cols / 2) / cols
            v = (k - rows / 2) / rows

            pi_v = math.pi * (u * a + v * b)
            if pi_v:
                mag = np.sin(pi_v) / pi_v
            else:
                mag = 1  # lim{x->0} sin(x)/x

            transformation[k, l] = mag * np.exp(complex(0, 1) * pi_v)

    return np.fft.fftshift(transformation)

def umbral(img,u):
    rs,cs = img.shape[:2]
    for r in range(rs):
        for c in range(cs):
            if img[r,c] < u:
                img[r, c] = 0
    return img

def quantum(img,level):
    M,N,c = img.shape
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    mask = "0"*(8-level) + "1"*level
    max_value = 0
    for i in range(M):
        for j in range(N):
            nBin = bin(img[i,j])[2:]
            nBin = nBin if len(nBin) == 8 else "0"*(8-len(nBin))+nBin

            img[i, j] = int(int(mask[0])*int(nBin[0])*math.pow(2,7)) + \
                        int(int(mask[1])*int(nBin[1])*math.pow(2,6)) + \
                        int(int(mask[2])*int(nBin[2])*math.pow(2,5)) + \
                        int(int(mask[3])*int(nBin[3])*math.pow(2,4)) + \
                        int(int(mask[4])*int(nBin[4])*math.pow(2,3)) + \
                        int(int(mask[5])*int(nBin[5])*math.pow(2,2)) + \
                        int(int(mask[6])*int(nBin[6])*math.pow(2,1)) + \
                        int(int(mask[7])*int(nBin[7])*math.pow(2,0))
            if img[i, j] > max_value:
                max_value = img[i, j]
    img = (img/max_value)*255


    return img

def get_intensity_line(image,d,parallel_axis):
    """
    Segun el eje de referencia y una distancia se obtiene una recta que contiene los niveles de intesidades de una imagen.
    image: Image dont care about if color or not.
    d: parallel distance from paralel_axis
    parallel_axis:
        It can be 0 -> X Axis : Horizontal
        It can be 1 -> Y Axis : Vertical
    """

    # image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    line = []
    if parallel_axis == 0: #Horizontal
        line = image[d,:]
    else: # Vertical
        line = image[:,d]

    return line

def gravity_object_1D(line):
    "Get center of gravity of a countinuous line that contains only white pixeles  in a general line with black pixeles."


    if (line.ndim == 1):
        centers = []
        i = 0

        while ( i < len(line)):

            # That mean we have a white pixel
            if line[i] > 128:
                index_init = i
                while i < len(line) and line[i] > 128:
                    i+= 1
                index_last = i
                centers.append(int(0.5*(index_init+index_last)))

            i += 1

        return centers
    else:
        print "[gravity_object_1D]: Error al momento de obtener el centro de gravedad de una imagen con mas de un canal"
        exit(1)

def get_info_bottles(bottles,y_covers):
    """

    :param bottles: Image [np.array]
    :param y_covers: Level of covers (Tapas de las botellas)
    :return: List  percent of empty of bottles
    """

    percent_empty = []
    bottles = cv2.cvtColor(bottles,cv2.COLOR_RGB2GRAY)

    covers = get_intensity_line(bottles, y_covers, 0)
    centers_of_bottles = gravity_object_1D(covers)

    height_image,a = bottles.shape

    print "Centros de las botellas en [x]: ", centers_of_bottles
    index_bottle = 1
    for cx in centers_of_bottles:

        bottle_line_center = get_intensity_line(bottles,cx,1)
        fill_bottle = gravity_object_1D(bottle_line_center)

        fill_bottle = fill_bottle[0]

        init = fill_bottle - 0.5*fill_bottle
        final = fill_bottle + 0.5*fill_bottle

        height_bottle = height_image-init
        height_empty = final - init


        percent_empty.append(height_empty*100/height_bottle)


        index_bottle += 1

    return percent_empty

def noisy(noise_typ,image):
    """
    The Function adds gaussian , salt-pepper , poisson and speckle noise in an image

    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.

    
    """
    if noise_typ == "gauss":
        row,col= image.shape[:2]
        # print "canales: " + str(ch)
        mean = 0
        var = 50
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        plt.subplot(121)
        plt.imshow(gauss,cmap='gray')

        gauss = np.float32(gauss)
        image = np.float32(image)
        image += gauss
        # plt.subplot(122)
        # plt.imshow(image,cmap='gray')

        # plt.figure("Histogramas")
        # plt.subplot(121)
        # histNOISE = cv2.calcHist([gauss],[0],None,[256],[-256,256])
        # plt.stem(range(int(-0.5*len(histNOISE)),int(0.5*len(histNOISE))),histNOISE,markerfmt='')
        # plt.subplot(122)
        # histIMAGENOISE = cv2.calcHist([image],[0],None,[256],[0,256])
        # plt.stem(range(len(histIMAGENOISE)),histIMAGENOISE,markerfmt='')
    
        # plt.show()  
        image = np.uint8(image)

        return image
    elif noise_typ == "s&p" or noise_typ == "s" or noise_typ == "p" :
        row,col = image.shape[:2]
        #Relacion Sal vs Pimienta
        s_vs_p = 0.5
        salt_value = 1
        pepper_value = 255

        #Porcentaje
        amount = 0.05
        out = np.copy(image)
        if "s" in noise_typ:
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
            out[coords] = pepper_value
        if "p" in noise_typ:
            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
            out[coords] = salt_value
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
    else:
        raise ValueError("No seas boludo, metiste cualquier nombre")

        return noisy

def media_aritmetica(img,m,n):
    kernel = np.ones((m, n), np.float32) / (m * n)
    return cv2.filter2D(img, -1, kernel)

def media_armonica(img,m,n):
    img_aux = img.copy()
    img_aux[np.where(img_aux==0)]=0.1
    img_aux = 1/img_aux
    kernel = np.ones((m, n), np.float32) / (m * n)
    result =cv2.filter2D(img_aux, -1, kernel)
    result[np.where(result==0)]=0.1
    result = 1/result
    result *= m*m*n*n
    return result

def media_geometrica(img,m,n):

    img[np.where(img == 0)] = 1

    return  np.uint8(
                np.exp(
                    cv2.boxFilter(
                        np.log(
                            np.float32(img)
                        ), -1, (m, n)
                    )
                )
            )

def media_contraArmonica(img,m,n,Q):

    imgHere = img.astype('float64')

    
    imgQ1 = np.power(imgHere,Q+1)
    imgQ = np.power(imgHere,Q)


    kernel = np.ones((m, n), np.float32) 
    
    resultQ1 = cv2.filter2D(imgQ1, -1, kernel)
    resultQ  = cv2.filter2D(imgQ, -1, kernel)

    resultQ[np.where(resultQ==0)]=0.1

    result = np.uint8(resultQ1/resultQ)

    return result

def generarImagenPatron():
    M = np.ones((300,300), np.uint8)*10
    M[50:250, 50:250] = np.ones((200,200), np.uint8)*130
    M = cv2.circle(M, (150, 150), 70, 230, -1)
    return M

def gr_gaussiano(img,mean, sigma):
    # img: imagen a la cual se le agrega ruido
    # mu: media
    # sigma: desviacion estandar
    gauss = np.random.normal(mean,sigma,img.shape)
    gauss = np.float32(gauss)

    return gauss

def gr_rayleigh(img, a):
    (alto, ancho) = img.shape
    img_r = cv2.normalize(img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ruido = np.random.rayleigh(a, [alto, ancho]).astype('f')
    img_r = img_r + ruido
    img_r = cv2.normalize(img_r, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_r

def gr_uniforme(img, a, b):
    (alto, ancho) = img.shape
    img_r = cv2.normalize(img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ruido = np.random.uniform(a, b, [alto, ancho]).astype('f')
    img_r = img_r + ruido
    img_r = cv2.normalize(img_r, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_r

def gr_exponencial(img, a):
    (alto, ancho) = img.shape
    img_r = cv2.normalize(img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ruido = np.random.exponential(a, [alto, ancho]).astype('f')
    img_r = img_r + ruido
    img_r = cv2.normalize(img_r, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_r

def gr_gamma(img, a, b):
    (alto, ancho) = img.shape
    img_r = cv2.normalize(img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    ruido = np.random.gamma(a, b, [alto, ancho]).astype('f')
    img_r = img_r + ruido
    img_r = cv2.normalize(img_r, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_r

def gr_salPimienta(img, s_vs_p, cantidad):
    # Parametros de entrada
    # img: imagen
    # s_vs_p: relacion de sal y pimienta [0, 1]
    # cantidad: cantidad de ruido en porcentaje [0, 1]
    imgWithNoise = img.copy()
    # Funcion para ensuciar una imagen con ruido sal y pimienta

    # generar ruido tipo sal
    n_sal = np.ceil(cantidad * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(n_sal)) for i in img.shape]

    rango = range(len(coords[0]))

    for i in rango:
        imgWithNoise[coords[0][i],coords[1][i]]= 255

    # img[coords] = np.ones(coords.shape)*255
    # generar ruido tipo pimienta
    n_pim = np.ceil(cantidad * img.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(n_pim)) for i in img.shape]
    rango = range(len(coords[0]))
    for i in rango:
        imgWithNoise[coords[0][i],coords[1][i]]= 0

    return imgWithNoise



def orderStatistcFilter(img,ksize,func):
    """
    img: Source
    ksize: Odd number
    func: np.median , max , min
    """
    
    
    k = ksize//2
    
    result = np.zeros(img.shape[:2])

    M,N = img.shape[:2]
    for i in range(M):
        for j in range(N):
            leftInd  = j-k -1
            rightInd = j +k +1
            upInd = i-k-1
            buttomInd = i+k+1

            if upInd < 0:
                upInd = 0
            if leftInd < 0:
                leftInd = 0
            if rightInd > N:
                rightInd = N
            if buttomInd > M:
                buttomInd = M
            
            resultFunc = func(img[upInd: buttomInd, leftInd : rightInd])                                
            result[i,j] = resultFunc

    return result

def filtro_img(img,filtro_magnitud):
    # Filtro para imagenes de un canal

    #como la fase del filtro es 0 la conversion de polar a cartesiano es directa (magnitud->x, fase->y)
    filtro=np.array([filtro_magnitud,np.zeros(filtro_magnitud.shape)]).swapaxes(0,2).swapaxes(0,1)
    
    plt.subplot(221)
    plt.title('Espectro del filtro')
    plt.imshow(filtro_magnitud,cmap='gray')
    

    imgf=cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    imgf  = np.fft.fftshift(imgf)
    plt.colorbar()
    plt.subplot(222)
    plt.title('Especto de la imagen [Centrada]')
    plt.imshow(20*np.log(np.abs(imgf[:,:,0])),cmap='gray')



    resultFH =cv2.mulSpectrums(imgf, np.float32(filtro), cv2.DFT_ROWS)
    plt.colorbar()
    plt.subplot(223)
    plt.title('Multiplicacion')
    plt.imshow(np.log(np.abs(resultFH[:,:,0])),cmap='gray')
    resultFH  = np.fft.fftshift(resultFH)
    result_fg = cv2.idft(resultFH, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    plt.colorbar()
    plt.subplot(224)
    plt.title('Resultado del producto')
    plt.imshow(np.abs(result_fg),cmap='gray')
    
    return result_fg

def dist(a,b):
    # distancia euclidea
    return np.linalg.norm(np.array(a)-np.array(b))

def filtro_gaussiano(rows,cols,corte):
    # Filtro de magnitud gaussiano

    magnitud = np.zeros((rows, cols))


    for k in range(rows):
        for l in range(cols):
            magnitud[k,l]=np.exp(-dist([k,l],[rows//2,cols//2])/2/corte/corte)

    return np.fft.ifftshift(magnitud)

def circle(M,N,radius, deltaX = 0, deltaY = 0) :

 

    n = min(M,N)
    a = int(M / 2) #y
    b = int(N / 2)
    r = radius

    y, x = np.ogrid[-a:M - a, -b:N - b]
    mask = (x - deltaX)*( x - deltaX) + (y -deltaY) * (y - deltaY)  <= r * r
    
    if (deltaY or deltaX):
        mask += (x + deltaX)*( x + deltaX) + (y +deltaY) * (y + deltaY)  <= r * r

    array = np.zeros((M, N))
    array[mask] = 1
    return  array

def filtro_ideal(rows, cols, corte,dx=0,dy=0,PasaBanda=True):
    # Filtro de magnitud ideal
    magnitud = np.zeros((rows, cols))
    # magnitud = cv.circle(magnitud, (cols//2 - dx, rows//2-dy), int(rows*corte), 1, -1)
    magnitud = circle(rows,cols,corte,dx,dy)
    if (not(PasaBanda) ):
        I = np.ones(magnitud.shape)
        magnitud = I - magnitud 

    return np.fft.ifftshift(magnitud)


def filtro_butterworth(rows, cols, corte,  order,PasaBanda = True,deltaP=[[0,0]]):
    """ 
    Filtro de magnitud Butterworth corte = D0 en imagen de lado 1
    1 / 1 + {D / Do}^{2n}
    corte: [Pixeles]
    """
    print " ------------------------------------------------------------ "
    print "     Generando filtro de Butterworth, con ", len(deltaP), " puntos."
    print " ------------------------------------------------------------ "
    print  ""

    Do = corte
    H = np.ones((rows, cols))

   

    for k in range(rows):
        for l in range(cols):
            # Segun la cantidad de centros, pregunto y luego hago la productoria
            # por defecto esta un punto en el centro, lo que me da la posibilidad de generalizar la funcion y obtener filtros en el origen
            for i in range(len(deltaP)):
                Duv  = np.linalg.norm(np.array([k,l])-np.array([rows//2 + deltaP[i][1], cols//2 + deltaP[i][0]]))            
                Duv_ = np.linalg.norm(np.array([k,l])-np.array([rows//2 - deltaP[i][1], cols//2 - deltaP[i][0]]))

                H[k,l] *= (1.0/(1 + (Do/Duv)**(2*order) )) * (1.0/(1 + (Do/Duv_)**(2*order) )) 

    if PasaBanda:
        H = 1 - H
            
    # newH = cv2.multiply(H,H_)
    # newH = H + H_
    # newH = cv2.normalize(newH, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    # return np.fft.ifftshift(magnitud)
    return H


def motion_blur(size, a,  b):
    # Filtro de movimiento en direcciones a y b
    transformation =np.zeros(size)
    rows = size[0]
    cols = size[1]

    #fase exp(j\pi (ua + vb))
    #magnitud \frac{ \sin(\pi(ua+vb)) }{ \pi (ua+vb) }
    for k in range(rows):
        for l in range(cols):
            u = (l-cols/2)/cols
            v = (k-rows/2)/rows

            pi_v = math.pi*(u*a+v*b)
            if pi_v:
                mag = np.sin(pi_v)/pi_v
            else:
                mag=1 #lim{x->0} sin(x)/x

            transformation[k,l] = mag*np.exp(complex(0,1)*pi_v)


    return np.fft.fftshift(transformation)





#--------------------- SEGMENTACION | CAPITULO 10  | TP 7 ---------------------

def deteccionPuntos(img):

    D2 = np.array(
        [
            [1,  1, 1],
            [1, -8, 1],
            [1,  1, 1]

        ]
    )
    return cv2.filter2D(img, -1, D2)

def deteccionLineas(img,ksize, grad= 0):
    "Detecta lineas de un pixel de grosor en la direccion segun grad (Grados)"

    mask = np.array(
        [
            [-1, -1, -1],
            [ 2,  2,  2],
            [-1, -1, -1]

        ]
    )

    # mask_r = rotate(mask,grad)
    mask_r = rotar(mask,grad)
    # mask_r = np.rot90(mask)   #rotate(mask,grad)

    plt.figure("Mascara para las lineas")
    plt.imshow(mask_r,cmap='gray')
    plt.show()

    result = cv2.filter2D(img,-1,mask_r)

    return result

def bordesG_1_derivada(img):
    Gx = np.array(
        [
            [ 0 , 0 , 0],
            [ 0 , -1, 0],
            [ 0 , 1 , 0]

        ]
    )
    Gy = np.array(
        [
            [0,   0,  0],
            [0,  -1,  1],
            [0,   0,  0]

        ]
    )
    bordes_x = cv2.filter2D(img,-1,Gx)
    bordes_y = cv2.filter2D(img,-1,Gy)

    return bordes_x, bordes_y

def bordesG_Roberts(img):
    Gx = np.array(
        [
            [ 0 , 0 , 0],
            [ 0 , -1, 0],
            [ 0 , 0 , 1]

        ]
    )
    Gy = np.array(
        [
            [0,  0,  0],
            [0,  0, -1],
            [0,  1,  0]

        ]
    )
    bordes_x = cv2.filter2D(img,-1,Gx)
    bordes_y = cv2.filter2D(img,-1,Gy)

    return bordes_x,bordes_y

def bordes_Lapla (img):
    D1 = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]

        ]
    )
    D2 = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]

        ]
    )

    return cv2.filter2D(img, -1, D2)

def bordes_LoG(img):
    D1 = np.array(
        [
            [0, 0, -1, 0 ,0],
            [0, -1, -2, -1 ,0],
            [-1, -2, 16, -2 ,-1],
            [0, -1, -2, -1 ,0],
            [0, 0, -1, 0 ,0]


        ]
    )


    return cv2.filter2D(img, -1, D1)
    # return filterImg(img,Gx) + filterImg(img,Gy)

def bordesG_Prewitt(img):
    Gx = np.zeros((3,3))
    Gy = np.zeros((3,3))
    Gx[0,:] = -1
    Gx[2,:] =  1

    Gy[:, 0] = -1
    Gy[:, 2] = 1

    bordes_x = cv2.filter2D(img,-1,Gx)
    bordes_y = cv2.filter2D(img,-1,Gy)

    return bordes_x,bordes_y

def bordesG_Sobel(img):
    Gx = np.zeros((3,3))
    Gy = np.zeros((3,3))
    Gx[0,:] = -1
    Gx[2,:] =  1
    Gx[0, 1] = -2
    Gy[:, 0] = -1
    Gy[:, 2] = 1
    Gx[2, 1] = 2

    bordes_x = cv2.filter2D(img,-1,Gx)
    bordes_y = cv2.filter2D(img,-1,Gy)

    return bordes_x,bordes_y

def hough_Transform(img,threshold,thita_i = None,thita_f = None):
    """ 
                                        Transformada de Hough
    Esta funcion ademas de calcular la transformada de Hough se puede establecer un rango de acumuladores, 
    como tambien el angulo aproximado de lineas que se desee detectar.
    -------------------------------------------------------------------------------------------------------
    Tener en mente que :
        thita = [0 , 2 pi] y rho > 0 | hasta D , donde D es la distancia diagonal de la imagen (en pixeles)
        Y thita crece desde el eje x
    
    """
    # Color de las lineas RGB
    COLOR = (255,0,0)
    # Lineas detectadas
    linesP = []
    
    # Umbral min and max
    bordes = cv2.Canny(img, 175, 225, apertureSize=3)
    imgWithLines = img.copy()
    imgWithLines = cv2.cvtColor(imgWithLines,cv2.COLOR_GRAY2RGB)

    cv2.imshow("Debug: Canny", bordes)

    if thita_i != None and thita_f != None:
        "Hough necesita los angulos en radianes"
        thita_i = (thita_i * np.pi )/ 180
        thita_f = (thita_f * np.pi )/ 180        
                            # IMG | Resol p | Resol Thita | 
        lines = cv2.HoughLines(bordes, 1, np.pi/180, threshold,min_theta=thita_i,max_theta=thita_f)
    else:
        lines = cv2.HoughLines(bordes, 1, np.pi/180, threshold)
    
    if (lines is None):
        totalLineas = 0
    else:
        totalLineas = len(lines)
        
        

    print "Total de lineas encontradas: ", totalLineas
    if totalLineas > 0:
        for line in lines:
            l = line[0] 

            rho = l[0]
            theta = l[1]
            # print theta *180 / np.pi

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            P1 = (x1, y1)
            P2 = (x2, y2)
            # Coordanas cartesianas
            # linesP.append([P1,P2])
            # Coordenadas Polares
            linesP.append( (rho,theta) )

            cv2.line(imgWithLines, P1, P2, COLOR, 1)
        

    else:
        print "No se han encontrado lineas segun los parametros especificados."

    return imgWithLines, linesP



def skeleton(self,img):
            
    kSize = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kSize,kSize))
    size = np.size(img)
    skeleton = np.zeros(img.shape,np.uint8)
    
    
    
    done = False

    k = 0
    plt.ion()
    while( not done):
        # First term (A (-) B)
        eroded = cv2.erode(img,kernel)
        # Second term (A (-) B ) o B
        eroded_opening = cv2.morphologyEx(eroded,cv2.MORPH_OPEN,kernel)
        # temp = cv2.dilate(eroded,kernel) 
        # Skeleton of iteration k -> Sk(A)
        # (A (-) B) - (A (-) B ) o B
        Sk = cv2.subtract(eroded,eroded_opening)
        # S(A) = U Sk(A)
        skeleton = cv2.bitwise_or(skeleton,Sk)

        # --- Interactive graphics
        plt.figure(0)
        plt.subplot(311),plt.imshow(eroded,cmap="gray"),plt.title("Eroded")
        plt.subplot(312),plt.imshow(eroded_opening,cmap="gray"),plt.title("Eroded and Opening")
        plt.subplot(313),plt.imshow(Sk,cmap="gray"),plt.title("Skeleton k ")
        plt.figure(5)
        plt.subplot(111),plt.imshow(skeleton,cmap="gray"),plt.title("Skeleton k = "  + str(k))
        
        plt.pause(3)
        plt.clf()

        img = eroded.copy()
                
        zeros = size - cv2.countNonZero(eroded)
        if zeros==size:
            done = True
        k +=1
        
            
    return skeleton

def convexHull(self,img):
        
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    # create an empty black image
    drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)
    
    return drawing , cv2.fillPoly(drawing, pts =hull, color=(255,255,255))

def masking(img,mask):

    nMask = cv2.normalize(mask, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    C1,C2,C3 = cv2.split(img)

    NC1 = cv2.multiply(nMask,C1)
    NC2 = cv2.multiply(nMask,C2)
    NC3 = cv2.multiply(nMask,C3)

    newImg = cv2.merge((NC1,NC2,NC3))

    return newImg


def borderClearing(self,image,mask):

    I = self.frame(image,10)


    
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    
    Reconstruction_by_dilation = self.morphologicalReconstructionbyDilation(I,mask,se,40)
    
    
    result = self.difference(mask,Reconstruction_by_dilation)

    return result

def difference(self,A,B):
    if (A.shape == B.shape):
        
        rows, cols= A.shape
        result = np.zeros((rows,cols),np.uint8)
        for  i in range(rows):
            for j in range(cols):
                if (A[i,j] and not(B[i,j])):
                    result[i,j]=1
        return result

def morphologicalReconstructionbyErotion(self,F,G,se,k):
    if (k == 0):
        return F
    if (k == 1):
        erotion_F_and_SE = opencv.erotion(F,se) 
        RD1 =   opencv.bitwise_or( erotion_F_and_SE, G)
        return RD1
    
    RDk_1 =  self.morphologicalReconstructionbyErotion(F,G,se,k-1)
    erotion_RDK1_and_SE = opencv.erotion(RDk_1,se) 
    RDk =   opencv.bitwise_or( erotion_RDK1_and_SE, G)
    if (np.array_equal(RDk,RDk_1)):
        
        print "MR by Erotion finished with ", k , " iterations."
        
        
        return RDk

    else:
        return RDk

def morphologicalReconstructionbyDilation(self,F,G,se,k):
    if (k == 0):
        return F
    if (k == 1):
        dilation_F_and_SE = opencv.dilate(F,se) 
        R_D_1 =   opencv.bitwise_and( dilation_F_and_SE, G)
        return R_D_1
    
    RDk_1 =  self.morphologicalReconstructionbyDilation(F,G,se,k-1)
    dilation_RDK1_and_SE = opencv.dilate(RDk_1,se) 
    RDk =   opencv.bitwise_and( dilation_RDK1_and_SE, G)
    if (np.array_equal(RDk,RDk_1)):
        
        print "MR by Dilation finished with ", k , " iterations."
        
        
        return RDk

    else:
        return RDk

def frame(self, image,frame_width):
    try:
        rows,cols ,_ = image.shape
        

        I = np.zeros((rows,cols),np.uint8)

        I[1:1+frame_width,:] = np.ones((frame_width,cols),np.uint8)
        I[rows-frame_width:rows ,:] = np.ones((frame_width,cols),np.uint8)


        I[:,1:1+frame_width] = np.ones((rows,frame_width),np.uint8)
        I[:  ,cols-frame_width:cols] = np.ones((rows,frame_width),np.uint8)

        return I
    except:
        print "Error in frame [function]"

def extractionConnectedComponnet(self, seed , mask, SE=[]):
    """
    Extraction of connected components
    **********************************
    seed: Matrix (NxM) that contains almost a pixel in the component
    mask: Matrix (NxM) that contains the original figure, and is the mask when aplicating growing (dilation)
    SE: 
    X_k = [X_(k-1) (+) B] \interseccion 

    """

    kernel = SE
    if (kernel == []):
        kernel = opencv.getStructuringElement(opencv.MORPH_RECT,(3,3))
    X_k_1 = seed
    
    result = opencv.dilate(X_k_1,kernel)
    X_k = opencv.bitwise_and(result,mask)
    k_max = 1000
    k = 0
    
    while(not(np.array_equal(X_k,X_k_1)) and k < k_max):
        X_k_1 = X_k
        k+=1
        result = opencv.dilate(X_k_1,kernel)
        X_k = opencv.bitwise_and(result,mask)
    print "extractionConnectedComponnet: INFO - Return with " + str(k)+ " iterations."
    return X_k

def toBinary(self, A ,threshold=127):
    # A = np.asarray(A,dtype=np.int)
    # return opencv.threshold(A,thresh=threshold, maxval=1)
    # retval, threshold = cv2.threshold(A, 127, 255, cv2.THRESH_BINARY)
    # retval, threshold = opencv.threshold(A,threshold,  255, opencv.THRESH_BINARY)
    print threshold

def invertColor(self, img):
    rows,cols = img.shape
    copyImage = img
    # print img.shape
    for i in range(rows):
        for j in range(cols):
            copyImage[i,j] = -img[i,j]+255
    
    return copyImage

def pegar(imgA,mask,imgB):
    """
    Recibe dos imagenes A y B con sus respectivos 3 canales
      además de una mascara que sera la que va a recortar 
        la imagen B y la va a pegar sobre la imagen A
    
    """
    # Separo las imagenes en sus respectivos canales
    A1,A2,A3 = cv2.split(imgA)    
    B1,B2,B3 = cv2.split(imgB)


    # Normalizo los canales para realizar la multiplicacion
    A1 = np.uint8(A1)
    A2 = np.uint8(A2)
    A3 = np.uint8(A3)
    mask = np.uint8(mask)
    B1 = np.uint8(B1)
    B2 = np.uint8(B2)
    B3 = np.uint8(B3)

    mask = cv2.normalize(mask, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    C1 = cv2.multiply(mask,B1)
    C2 = cv2.multiply(mask,B2)
    C3 = cv2.multiply(mask,B3)
    
    
    mask_inv = cv2.bitwise_not(mask)
    mask_inv = cv2.normalize(mask_inv, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    D1 = cv2.multiply(mask_inv,A1)
    D2 = cv2.multiply(mask_inv,A2)
    D3 = cv2.multiply(mask_inv,A3)

    E1 = C1 + D1
    E2 = C2 + D2
    E3 = C3 + D3

    result = cv2.merge((E1,E2,E3))

    return result

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

    mask = segmentador(img,umbralCanalesMinimos,umbralCanalesMaximos)

    return mask

def stadistics(img):
    "Funcion utilizada para el filtro adaptativo "

    mean = np.mean(img)
    variance = np.var(img)
    
    return mean, variance

def ALNRFilter(kernel,gVariance=20.0):
    "Adaptive, Local Noise Reduction Filter"
    M,N = kernel.shape[:2]
    fxy = 0
    if M != 0 and N != 0:
        mL, oL = stadistics(kernel)

    
        gxy = kernel[int(0.5*M),int(0.5*N)]
        constVar = gVariance/oL
        fxy = gxy - constVar*(gxy-mL)
    
    return fxy

def midpointFilter(kernel):
    nMin = float(np.min(kernel))
    nMax = float(np.max(kernel))

    midpoint =0.5*(nMax+nMin)

    return midpoint

def alphaTrimmedFilter(kernel,d=2):
    sortedList= np.sort(kernel.ravel())
    
    return np.mean(sortedList[d:-d])
    def media_no_lineal(self,img,ksize,func):
        """
        img: Source
        ksize: Odd number
        func: np.median , max , min
        """
        
        
        k = ksize//2
        
        result = np.zeros(img.shape[:2])

        M,N = img.shape[:2]
        for i in range(M):
            for j in range(N):
                leftInd  = i-k
                rightInd = i +k
                upInd = j-k
                buttomInd = j+k

                if upInd < 0:
                    upInd = 0
                if leftInd < 0:
                    leftInd = 0
                if rightInd > N:
                    rightInd = N
                if buttomInd > M:
                    buttomInd = M
                
                resultFunc = func(img[upInd: buttomInd, leftInd : rightInd])                                
                result[i,j] = resultFunc

        return result

def maximosCompFrecuenciales(spectrum,umbral,Dmin = 1):
    "Recibo el espectro de la imagen (centrada) y retorno los componentes frecuenciales maximos "

    _, img = cv2.threshold(spectrum,umbral,1,cv2.THRESH_BINARY)
    plt.figure("Detector de componentes frecuenciales maximas")
    plt.imshow(img,cmap='gray')
    plt.show()
    result = np.where(img == np.amax(img))
    M,N = spectrum.shape[:2]
    Cx = np.ones(result[0].shape)*int(N*0.5)
    Cy = np.ones(result[0].shape)*int(M*0.5)

    

    um = result[1] - Cy 
    vm = result[0] - Cx

    freqMaxVal = []
    for indx in range(len(um)):

        if ((um[indx] > 0 and vm[indx] > 0) or (um[indx] > 0 and vm[indx] == 0) or (um[indx] == 0 and vm[indx] > 0)) :
            D =  dist([um[indx],vm[indx]],[0,0])
            if (D > 4):
                freqMaxVal.append([um[indx],vm[indx]]) 
    


    listaOptima = puntosNoCercanos(freqMaxVal,Dmin)
    
    print "Puntos detectados ", listaOptima
    return listaOptima

def puntosNoCercanos(listaPuntos,D):

    puntosGood = []
    puntosAnalizados = listaPuntos
    i = 0

    while len(puntosAnalizados) > 0:
        p = puntosAnalizados[i]

        puntosAnalizados.remove(p)
        distancias =  map(lambda x : dist(p,x), puntosAnalizados )


        if len( filter(lambda x : x <= D,distancias ) ) == 0 :
            puntosGood.append(p)


            

    
    return puntosGood

def errorMedioCuadratico(img, img_):
    M,N = img.shape[:2]
    suma = 0
    for i in range(M):
        for j in range(N):
            suma += (img[i,j]- img_[i,j])**2

    meanSquareError = suma/(N*M)

    return meanSquareError
    
def compararHistogramas(imgA,imgB):

    plt.figure()
    plt.subplot(121)
    img1 = equalizarIMG(imgA)
    plt.imshow(img1)
    plt.subplot(122)
    img2 = equalizarIMG(imgB)
    plt.imshow(img2)
    plt.show()
    # Convert it to HSV
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # Se compara el Hue y Saturation. Luego se normaliza
    hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    print hist_img1
    hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # find the metric value
    # param= cv2.HISTCMP_CORREL
    # param= cv2.HISTCMP_CHISQR
    # param= cv2.HISTCMP_CHISQR_ALT
    # param= cv2.HISTCMP_INTERSECT
    param= cv2.HISTCMP_BHATTACHARYYA #Con este metodo, generalmente se considera diferencias entre 0 - 1.0, cuando > 0.75 se considera que existe mucha diferencia
    # param= cv2.HISTCMP_HELLINGER
    # param= cv2.HISTCMP_KL_DIV
    metric_val = cv2.compareHist(hist_img1, hist_img2, param)


    return metric_val

def equalizarIMG(img):

    C1,C2,C3 = cv2.split(img)

    eC1 = cv2.equalizeHist(C1)
    eC2 = cv2.equalizeHist(C2)
    eC3 = cv2.equalizeHist(C3)

    eqIMG = cv2.merge((eC1,eC2,eC3))

    return eqIMG
def histograma3C(img):
    "Calcula y muestra el histograma para una imagen que contiene 3 canales"

    hisA = cv2.calcHist([img],[0],None,[256],[0,256])
    hisB = cv2.calcHist([img],[1],None,[256],[0,256])
    hisC = cv2.calcHist([img],[2],None,[256],[0,256])
    
    plt.plot(range(len(hisA)), hisA,'r')
    plt.plot(range(len(hisB)), hisB,'g')
    plt.plot(range(len(hisC)), hisC,'b')

    return hisA,hisB,hisC


def infoSeg(mask,threshold):
    M,N, = mask.shape[:2]
    countSeg =  (mask > 0).sum()
    porcentaje = float(countSeg)/float(M*N)
    return porcentaje > threshold 