
# GUIA PRACTICA N7 - NOCIONES DE SEGMENTACION


from  pdifunFixed import *
from matplotlib import pyplot as plt

class TP7:
    __basePath = "img/TP7 - Segmentacion/"

    def ejercicio1(self, letra):
        img = cv2.imread(self.__basePath, 0)
        




if __name__ == '__main__':

    img = cv2.imread('img/snowman.png',0)
    img = noisy("gauss",img)
    plt.subplot(121)
    plt.imshow(hough_Transform(img,50,13,16), interpolation='nearest', cmap='gray')
    plt.title('Border Detection with LoG')

    plt.subplot(122)
    plt.imshow(img, interpolation='nearest', cmap='gray')
    plt.title('Border Detection with LoG')

    plt.show()


    


    # # Make plot with vertical (default) colorbar
    # fig, ax = plt.subplots()
    #
    # # plt.subplot(131),plt.imshow(img, cmap = 'gray')
    # # plt.title('Imagen Original'), plt.xticks([]), plt.yticks([])
    #
    # plt.subplot(131)
    # plt.imshow(bordes_Prewitt(img), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with Prewitt')
    #
    # plt.subplot(132)
    # plt.imshow(bordes_Sobel(img), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with Sobel')
    #
    # plt.subplot(133)
    # plt.imshow(bordes_Roberts(img), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with Roberts')
    #
    #
    # plt.figure(2)
    #
    # plt.subplot()
    # plt.imshow(bordes_Lapla(img), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with Lapla')
    #
    # plt.figure(3)
    #
    # plt.subplot()
    # plt.imshow(bordes_LoG(img), interpolation='nearest', cmap='gray')
    # plt.title('Border Detection with LoG')
    #
    # plt.show()
    # # Add colorbar, make sure to specify tick locations to match desired ticklabels
    # # cbar = fig.colorbar(cax)
    #
    #
    # plt.show()
