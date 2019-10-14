import pdifunFixed as pdi
from matplotlib import pyplot as plt
import cv2 
import numpy as np

class TP7:
    """GUIA PRACTICA N7 - NOCIONES DE SEGMENTACION"""

    __basePath = "img/TP7 - Segmentacion/"
    
    def induccion(self):
        
        img = cv2.imread(self.__basePath+"mosquito.jpg", 0)

        self.lineDetection(img)
    


    def ejercicio1(self):
        img = cv2.imread(self.__basePath+"mosquito.jpg", 0)
        noise = pdi.gr_gaussiano(img,0,50)
        imgWithNoise = img + noise
        plt.figure("Ruido")
        plt.subplot(121)
        plt.imshow(img,cmap='gray')
        plt.subplot(122)
        imgWithNoise = cv2.normalize(imgWithNoise, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.imshow(imgWithNoise,cmap='gray')
        plt.show()
        # self.edgeDetectionAlls(img)
        
        self.edgeDetectionAlls(imgWithNoise)

    def ejercicio2(self):
        
        # nameIMG = "snowman.png"
        nameIMG = "letras3.tif"

        img = cv2.imread(self.__basePath+nameIMG ,0)
        
        plt.figure()
        plt.imshow(img, interpolation='nearest', cmap='gray')
        plt.title('Imagen Original')
        plt.figure()
        result,lineas = pdi.hough_Transform(img,50,0,2)
        print "Lineas -> ", lineas
        plt.title('Transformada de Hough')
        # cv2.line(result, (200,40), (165,100), (255,0,0), 1)
        # cv2.line(result, (0,0), (175,87), (255,0,0), 1)
        plt.imshow(result, interpolation='nearest', cmap='gray')
        

        plt.show()





    def lineDetection(self,img):
        grados = 0
        result = pdi.deteccionLineas(img,5, grados)
        _ ,result = cv2.threshold(result,250,255,cv2.THRESH_BINARY)

        plt.figure("Deteccion de lineas")
        plt.subplot(121),plt.title("Imagen original")
        plt.imshow(img,cmap='gray')
        plt.subplot(122),plt.title("Lineas de {} grados".format(grados))
        plt.imshow(result,cmap='gray')

        plt.show()

    def pointDetection(self,img):
        
        result = pdi.deteccionPuntos(img)
        _ ,result = cv2.threshold(result,200,255,cv2.THRESH_BINARY)        
        plt.figure("Deteccion de puntos")
        plt.subplot(121),plt.title("Imagen original")
        plt.imshow(img,cmap='gray')
        plt.subplot(122),plt.title("Puntos")
        plt.imshow(result,cmap='gray')

        plt.show()

    def edgeDetectionAlls(self, img):

        Gx,Gy    = pdi.bordesG_1_derivada(img)
        bPriDerivada = Gx+Gy
        Gx,Gy    = pdi.bordesG_Roberts(img)
        bRoberts = Gx+Gy
        Gx,Gy    = pdi.bordesG_Prewitt(img)
        bPrewitt = Gx+Gy
        Gx,Gy    = pdi.bordesG_Sobel(img)
        bSobel   = Gx+Gy
        Gxy      = pdi.bordes_Lapla(img)
        bLapla   = Gxy
        Gxy      = pdi.bordes_LoG(img)
        bLog     = Gxy


        plt.figure("Edge Detection")
        plt.subplot(231),plt.title("Primera Derivada")
        plt.imshow(bPriDerivada,cmap='gray')

        plt.subplot(232),plt.title("Roberts")
        plt.imshow(bRoberts,cmap='gray')

        plt.subplot(233),plt.title("Prewitt ")
        plt.imshow(bPrewitt,cmap='gray')

        plt.subplot(234),plt.title("Sobel")
        plt.imshow(bSobel,cmap='gray')

        plt.subplot(235),plt.title("Lapla ")
        plt.imshow(bLapla,cmap='gray')

        plt.subplot(236),plt.title("LoG ")
        plt.imshow(bLog,cmap='gray')


        plt.show()




    def probandoCanny(self,img):
        # PRUEBA DE BORDES CON CANNY
        bordes = cv2.Canny(img,100,200)
    
        contornos, _  = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(img, contornos,-1,(0,0,255),2)

        print "Contornos encontrados : ", len(contornos)


        plt.figure("Flow")
        plt.subplot(121)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)

        plt.subplot(122)
        plt.imshow(bordes,cmap='gray')

        plt.show()




def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    return out

global seed 

def region_growing(img, seed):

    outimg = np.zeros_like(img)

    if (len(seed)>0):
        c = 0
        for iseed in seed:
            print "Procesando semilla: {}".format(c)
            c += 1
            list = []   
            list.append((iseed[0], iseed[1]))
            processed = []

            while(len(list) > 0):
                pix = list[0]
                outimg[pix[0], pix[1]] = 255
                for coord in get8n(pix[0], pix[1], img.shape):
                    if img[coord[0], coord[1]] != 0:
                        outimg[coord[0], coord[1]] = 255
                        if not coord in processed:
                            list.append(coord)
                        processed.append(coord)
                list.pop(0)
            
    else:
        print "No hay semillas!"

    return outimg

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print 'Seed: ' + str(x) + ', ' + str(y), img[y,x]
        clicks.append((y,x))
        if (len(clicks) == nSeed):
            cv2.destroyWindow('Input')



def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print 'x = %d, y = %d'%( ix, iy)


    seed=[( int(ix), int(iy) )]

    if len(seed) == 1:
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)

    return seed



if __name__ == '__main__':

    tp7 = TP7()

    # tp7.ejercicio1()
    # tp7.induccion()
    # tp7.ejercicio2()




    # clicks = []
    image = plt.imread('letras1.tif',0)
    # image2 = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    _,img = cv2.threshold(image,128,1,cv2.THRESH_BINARY)
    # fig =  plt.figure("Imagen a procesar")
    # plt.imshow(image2)
    # plt.show()


    plt.figure("Imagen a procesar")
    plt.imshow(img,cmap='gray')
    plt.show()
    
    seed= [(40,100),(190,100)]
    result = region_growing(img,seed)

    plt.figure("Resultado")
    plt.imshow(result)
    plt.show()


    # seed = []

    # cid = fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show()

  
    
    # 





