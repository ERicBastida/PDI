#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"

using namespace cv;
using namespace pdi;
using namespace std;

int main(int argc, char** argv) {
//	Mat img = imread("corrientes_ruidogris.jpg",IMREAD_GRAYSCALE);
	Mat img = imread("iguazu_ruidogris.jpg",IMREAD_GRAYSCALE);
	Mat imgfinal(img.size(),img.type());
	
	//Elimino ruido gaussiano y ruido sal con media alfa recortado
	filtro_mediaAlfa_recortado(img, 2, 5);	
//	imgfinal = img(Rect(3,3,img.cols-3,img.rows-3));
	
	// Binarizo la imagen-
	threshold(img,imgfinal,100,255,THRESH_BINARY);
	
	// Aplico erosion
	Mat ee = getStructuringElement(MORPH_RECT,Size(3,3));
	morphologyEx(imgfinal,imgfinal,MORPH_ERODE,ee);
	morphologyEx(imgfinal,imgfinal,MORPH_ERODE,ee);
	morphologyEx(imgfinal,imgfinal,MORPH_ERODE,ee);
	morphologyEx(imgfinal,imgfinal,MORPH_OPEN,ee);
		
	
	// Aplico Detector de lineas
	imgfinal = DetectorLineas(imgfinal,3);	
	
	// Aplico  Hough
	imgfinal = DibujarHough(imgfinal, 50);
	
	
//	
//	vector<int> rango(2), color(3);
//	rango[0] = 0;
//	rango[1] = 255;
//	
//	color[0] = 0;
//	color[1] = 0;
//	color[2] = 255;
//	PseudoColor(img, vector<int> &color, vector<int> &rango);
//	
	namedWindow("Imagen1");
	imshow("Imagen1",imgfinal);

	waitKey(0);
	return 0;
} 
