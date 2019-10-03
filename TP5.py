import cv2
import numpy as np
import  pdifunFixed as pdi
from matplotlib import pyplot as plt
import math
import cmath



def baseFourier(M,N,u,v):
    "Retorna una base de Fourier segun el tamanio y frecuencias en direccion x o y"

    basFourier = np.zeros((M,N),dtype=complex)



    for x in range(M):
        for y in range(N):
            Fij = 0
            arg = 2*math.pi*(float(u*x)/float(M) + float(v*y)/float(N) )
            Fij = cmath.exp(-0.j*arg)
            # Version Polar, segun la formula de Euler
            # Fij += math.cos(arg) 
            # Fij += - j*math.sin(arg)

            basFourier[x,y]= Fij

    return basFourier
def Fourier(img,u,v):

    M,N = img.shape[:2]
    Fuv = 0   

    for x in range(M):
        for y in range(N):  
            arg0 = float(u*x)/float(M)  
            arg1 = float(v*y)/float(N)
            arg = 2*math.pi*( arg0 + arg1 )
            R = math.cos(arg)
            I = -math.sin(arg)
            imgxy = complex(img[x,y],0)
            fouxy = complex(R,I)
            Fuv += imgxy*fouxy

    return Fuv



class TP5:
    __basePATH = "img/TP5 - Fourier/"

    def circle(self,M,N,radius, deltaX = 0, deltaY = 0) :


        n = min(M,N)
        a = int(M / 2) #y
        b = int(N / 2)
        r = radius

        y, x = np.ogrid[-a:M - a, -b:N - b]

        mask = x * x + y * y <= r * r

        array = np.zeros((M, N))
        array[mask] = 1
        return  array

    def induccion(self):

        #Tamanio de la imagen
        M = 100
        N = 100

        self.eje1a(M,N)

        img = cv2.imread(self.__basePATH+"moon.jpg",0)

        M,N = img.shape[:2]
        
        # for x in range(N):
        #     for y in range(M):
        #         img[x,y] = img[x,y] * math.exp(x+y)


        filtro = pdi.filterIdeal(M,N,0.2)


        resultado = pdi.filtro_img(img,filtro)
        # plt.figure("resultados")
        # plt.subplot(121)
        # plt.imshow(filtro,cmap='gray')
        # plt.subplot(122)
        # plt.title("Spectrum Patronus of Fourier Base ")
        # plt.imshow(resultado,cmap="gray")

        # plt.title("Phase -pi / pi")
        # plt.imshow(phase,cmap="gray")

        # plt.figure("Lo bueno")
        # # plt.subplot(121)
        # plt.title("Spectrum Patronus of Fourier Base u= "+str(u)+" v= "+str(v))
        # plt.imshow(spectrum2,cmap="gray")
        # # plt.subplot(122)/
        # # plt.title("Phase -pi / pi")
        # # plt.imshow(phase2,cmap="gray")

        # plt.show()

    def eje1a(self,N,M):

        line_width = 1
        quad_width = int(0.35*M)
        rectangle_width = int(0.1*M)
        rectangle_height = int(0.05*M)

        radius = 20

        Cx = int(M/2)
        Cy = int(N/2)


        self.linea_vertical = np.zeros((M,N))
        self.linea_vertical[:,Cx-line_width : Cx +line_width ] = 1

        self.linea_horizontal = np.zeros((M,N))
        self.linea_horizontal[Cy-line_width : Cy +line_width,:] = 1

        self.cuadrado = np.zeros((M,N))
        self.cuadrado[Cy-quad_width : Cy +quad_width,Cx-quad_width : Cx +quad_width] = 1


        self.rectangulo = np.zeros((M,N))
        self.rectangulo[Cy-rectangle_height : Cy +rectangle_height,Cx-rectangle_width : Cx +rectangle_width] = 1

        n = min(M,N)
        a = int(M / 2) #y
        b = int(N / 2)
        r = radius

        y, x = np.ogrid[-a:M - a, -b:N - b]

        mask = x * x + y * y <= r * r

        array = np.zeros((M, N))
        array[mask] = 1

        self.circulo = array

    def ejercicio1(self):

        #Construye las figuras pedidas en la consigna
        self.eje1a(256,256)
        
        imagen = pdi.rotate(self.linea_horizontal,10)

        print Fourier(imagen,0,0)

        plt.subplot(121)
        plt.imshow(imagen, cmap = 'gray')
        plt.title('Imagen')
        plt.xticks([]), plt.yticks([])
        
        plt.subplot(122),
        plt.imshow(pdi.spectrum(imagen), cmap = 'gray')
        plt.title('Espectrum Patronus')
        plt.xticks([]), plt.yticks([])

        plt.show()

    def ejercicio2(self):

        img = cv2.imread(self.__basePATH+"moon.jpg",0)
        M,N = img.shape[:2]

        plt.subplot(221)
        plt.imshow(img,cmap='gray')
        # filtro = pdi.filtro_ideal(M,N,0.2)
        # filtro = pdi.filtro_butterworth(M,N,0.2,20)
        filtro = pdi.filtro_gaussiaon(M,N,0.001)
        plt.subplot(222)
        # filtro2show= np.array([filtro,np.zeros(filtro.shape)]).swapaxes(0,2).swapaxes(0,1)
        plt.imshow(filtro,cmap='gray')


        result = pdi.filtro_img(img,filtro)

        plt.subplot(223)
        plt.imshow(result,cmap='gray')

        plt.subplot(224)
        spec = pdi.spectrum(img)


        plt.imshow(spec,cmap='gray')

        plt.show()

    def ejercicio4(self):


        img = cv2.imread(self.__basePATH+ "camaleon.tif",0)
        Mo, No = img.shape

        img = pdi.optimalDFTImg(img)
        M, N = img.shape


        filter = pdi.filterGaussian(M,N,0.001,PA=True)

        A = 100
        a = (A-1)

        b= 1

        H = np.ones(filter.shape) * a + filter*b


        resultado = pdi.filterImg(img,H)

        img = img[0:Mo,0:No]
        resultado = resultado[0:Mo,0:No]




        plt.subplot(221),plt.imshow(img, cmap = 'gray')
        plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])

        plt.subplot(222),plt.imshow(pdi.spectrum(img), cmap = 'gray')
        plt.title('DFT de la original '), plt.xticks([]), plt.yticks([])

        plt.subplot(223),plt.imshow(np.fft.ifftshift(H), cmap = 'gray')
        plt.title('Filtro Gaussiano'), plt.xticks([]), plt.yticks([])

        plt.subplot(224),plt.imshow(pdi.spectrum(H), cmap = 'gray')
        plt.title('DFT Filtro'), plt.xticks([]), plt.yticks([])


        plt.figure()
        plt.imshow(resultado, cmap='gray')

        plt.show()



        


        

tp5 = TP5()
# tp5.induccion()
# tp5.ejercicio1()
# tp5.ejercicio2()
tp5.ejercicio4()

            











# cv2.imshow("Linea",optimalDFTImg(rectangulo*255))
# cv2./waitKey(0)
#


# EJERCICIO 2 - FILTROS PASA-BAJO Y PASA-ALTOS

# img = cv2.imread("img/fruta.jpg",0)
# Mo, No = img.shape
#
# # img = np.float(img)
# img = optimalDFTImg(img)
# M, N = img.shape
#
# # filter = filterIdeal(M,N,0.05)
# # filter = filterButterworth(M,N,0.005,100)
# filter = filterGaussian(M,N,0.005)
#
# resultado = filterImg(img,filter)
#
# img = img[0:Mo,0:No]
# resultado = resultado[0:Mo,0:No]
#
#
#
#
# plt.subplot(131),plt.imshow(img, cmap = 'gray')
# plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(133),plt.imshow(resultado, cmap = 'gray')
# plt.title('Resultado '), plt.xticks([]), plt.yticks([])
#
# plt.subplot(132),plt.imshow(np.fft.ifftshift(filter), cmap = 'gray')
# plt.title('Filtro ideal'), plt.xticks([]), plt.yticks([])
#
#
# plt.show()
#
# Inciso 3
# img = cv2.imread("img/fruta.jpg",0)
# Mo, No = img.shape
#
# # img = np.float(img)
# img = optimalDFTImg(img)
# M, N = img.shape
#
#
# filter = filterGaussian(M,N,0.005)
#
# resultado = filterImg(img,filter)
#
# img = img[0:Mo,0:No]
# resultado = resultado[0:Mo,0:No]
#
#
#
#
# plt.subplot(221),plt.imshow(img, cmap = 'gray')
# plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(222),plt.imshow(spectrum(img), cmap = 'gray')
# plt.title('DFT '), plt.xticks([]), plt.yticks([])
#
# plt.subplot(223),plt.imshow(np.fft.ifftshift(filter), cmap = 'gray')
# plt.title('Filtro Gaussiano'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(224),plt.imshow(spectrum(filter), cmap = 'gray')
# plt.title('DFT Filtro'), plt.xticks([]), plt.yticks([])
#
# plt.show()

# Ejercicio 4

