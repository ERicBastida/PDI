#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

from __future__ import division
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv2
import math
from scipy import ndimage

def histograma(img):
    result = cv2.calcHist([img],[0],None,[256],[0,256])
    return result

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
    h = cv.getOptimalDFTSize(img.shape[0])
    w = cv.getOptimalDFTSize(img.shape[1])
    return cv.copyMakeBorder(img, 0, h - img.shape[0], 0, w - img.shape[1], cv.BORDER_CONSTANT)

def spectrum(img):
    """Calcula y muestra el modulo logartimico de la DFT de img."""
    # img=optimalDFTImg(img)

    imgf = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    modulo = np.log(cv.magnitude(imgf[:, :, 0], imgf[:, :, 1]) + 1)
    modulo = np.fft.fftshift(modulo)
    modulo = cv.normalize(modulo, modulo, 0, 1, cv.NORM_MINMAX)

    return modulo

def rotar(img,angle):
    return ndimage.rotate(img,angle)

def rotate(img, angle):
    """Rotacion de la imagen sobre el centro"""
    if (angle != 0):
        r = cv.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), angle, 1.0)
        result = cv.warpAffine(img, r, img.shape)
        return result
    else:
        return img

def filterImg(img, filtro_magnitud):
    """Filtro para im�genes de un canal"""

    # como la fase del filtro es 0 la conversi�n de polar a cartesiano es directa (magnitud->x, fase->y)
    filtro = np.array([filtro_magnitud, np.zeros(filtro_magnitud.shape)]).swapaxes(0, 2).swapaxes(0, 1)
    imgf = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)

    imgf = cv.mulSpectrums(imgf, np.float32(filtro), cv.DFT_ROWS)

    return cv.idft(imgf, flags=cv.DFT_REAL_OUTPUT | cv.DFT_SCALE)

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
    magnitud = cv.circle(magnitud, (cols // 2, rows // 2), int(rows * corte), 1, -1)
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
    img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
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

    # image = cv.cvtColor(image,cv.COLOR_RGB2GRAY)

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
    bottles = cv.cvtColor(bottles,cv.COLOR_RGB2GRAY)

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
    img_aux = 1/img_aux
    kernel = np.ones((m, n), np.float32) / (m * n)
    result =cv2.filter2D(img_aux, -1, kernel)
    result = 1/result
    result *= m*m*n*n
    return result

def media_geometrica(img,m,n):
    rs,cs = img.shape[:2]
    for r in range(rs):
        for c  in range(cs):
            if img[r,c] == 0:
                img[r, c] = 1
    return  np.uint8(np.exp(cv2.boxFilter(np.log(np.float32(img)), -1, (m, n))))

def generarImagenPatron():
    M = np.ones((300,300), np.uint8)*10
    M[50:250, 50:250] = np.ones((200,200), np.uint8)*130
    M = cv.circle(M, (150, 150), 70, 230, -1)
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
    img_r = cv.normalize(img, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    ruido = np.random.rayleigh(a, [alto, ancho]).astype('f')
    img_r = img_r + ruido
    img_r = cv.normalize(img_r, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return img_r

def gr_uniforme(img, a, b):
    (alto, ancho) = img.shape
    img_r = cv.normalize(img, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    ruido = np.random.uniform(a, b, [alto, ancho]).astype('f')
    img_r = img_r + ruido
    img_r = cv.normalize(img_r, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return img_r

def gr_exponencial(img, a):
    (alto, ancho) = img.shape
    img_r = cv.normalize(img, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    ruido = np.random.exponential(a, [alto, ancho]).astype('f')
    img_r = img_r + ruido
    img_r = cv.normalize(img_r, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return img_r

def gr_gamma(img, a, b):
    (alto, ancho) = img.shape
    img_r = cv.normalize(img, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    ruido = np.random.gamma(a, b, [alto, ancho]).astype('f')
    img_r = img_r + ruido
    img_r = cv.normalize(img_r, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
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

def filtroMediaGeometrica(img, m, n):
    (s, t) = img.shape
    img = np.float32(img)
    for i in range(0, s-m+1):
        for j in range(0, t-n+1):
            acum = 1
            for k in range(i, i+m):
                for o in range(j, j+n):
                    acum = acum * img[k, o]
            img[i,j] = float(pow(acum, 1.0/(m*n)))
    return np.uint8(img)

def filtroMediaContraarmonica(img, Q, s, t):
    (m, n) = img.shape
    img = np.float32(img)
    for i in range(0, m-s+1):
        for j in range(0, n-t+1):
            cont1 = 0
            cont2 = 0
            for k in range(i, i+s):
                for o in range(j, j+t):
                    cont1 = cont1 + np.power(img[k, o], Q+1)
                    cont2 = cont2 + np.power(img[k, o], Q)
            img[i, j] = cont1 / cont2
    return np.uint8(img)

def filtroMediaAlfaRecortada(img, s, t, d):
    (m, n) = img.shape
    for i in range(0, m-s):
        for j in range(0, n-t):
            cont = 0
            for k in range(i, i+s):
                for o in range(j, j+t):
                    cont = cont + img[k, o]

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

def filtro_img(img,filtro_magnitud):
    # Filtro para im�genes de un canal

    #como la fase del filtro es 0 la conversi�n de polar a cartesiano es directa (magnitud->x, fase->y)
    filtro=np.array([filtro_magnitud,np.zeros(filtro_magnitud.shape)]).swapaxes(0,2).swapaxes(0,1)
    
    plt.subplot(221)
    plt.title('Espectro del filtro')
    plt.imshow(filtro_magnitud,cmap='gray')

    imgf=cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)

    plt.subplot(222)
    plt.title('Filtro en el dominio espacial')
    plt.imshow(20*np.log(np.abs(imgf[:,:,0])),cmap='gray')

    resultFH =cv.mulSpectrums(imgf, np.float32(filtro), cv.DFT_ROWS)
    plt.subplot(223)
    plt.title('Multiplicacion')
    plt.imshow(np.log(np.abs(resultFH[:,:,0])),cmap='gray')
    result_fg = cv.idft(resultFH, flags=cv.DFT_REAL_OUTPUT | cv.DFT_SCALE)
    plt.subplot(224)
    plt.title('Resultado del producto')
    plt.imshow(np.abs(result_fg),cmap='gray')
    plt.show()
    return result_fg

def dist(a,b):
    # distancia euclidea
    return np.linalg.norm(np.array(a)-np.array(b))

def filtro_gaussiaon(rows,cols,corte):
    # Filtro de magnitud gaussiano

    magnitud = np.zeros((rows, cols))

    corte *= rows
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

def filtro_butterworth(rows, cols, corte, order):
    """ Filtro de magnitud Butterworth
    corte = w en imagen de lado 1
    1 \over 1 + {D \over w}^{2n}"""
    magnitud = np.zeros((rows, cols))
    corte *= rows
    for k in range(rows):
        for l in range(cols):
            d2 = dist([k,l],[rows//2,cols//2])
            magnitud[k,l] = 1.0/(1 + (d2/corte)**(order+order))

    return np.fft.ifftshift(magnitud)

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
        Y thita crece desde el eje y
    
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