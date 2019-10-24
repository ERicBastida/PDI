#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"
#include <cmath>
#include <opencv2/gpu/gpu.hpp>


using namespace cv;
using namespace pdi;
using namespace std;

#define PI 3.14159265

Point2d LimSuperior(Mat img){
	Point2d LS;
	for(int i=0; i<img.rows;i++){
		for(int j=0; j<img.cols;j++){
			int v = (int)img.at<uchar>(i,j);
			if ( v == 0){
				LS.x = j;
				LS.y = i;
				return LS;
			}
		}	
	}
}
Point2d LimInferior(Mat img){
	Point2d LI;
	for(int i=img.rows-1; i>0;i--){
		for(int j=img.cols-1; j>0;j--){
			int v = (int)img.at<uchar>(i,j);
			if ( v == 0){
				LI.x = j;
				LI.y = i;
				return LI;
			}
		}	
	}
}

int main(int argc, char** argv) {
	Mat img = imread("escaneo1.jpg");
	info(img);
	
	/** ************************************************* **/
	/**        Paso a escalas de gris y umbralizo         **/
	/** ************************************************* **/
	Mat imgBin;
	cvtColor(img,imgBin,COLOR_BGR2GRAY);
	threshold(imgBin,imgBin,200,255,THRESH_BINARY_INV);
	
	/** ************************************************* **/
	/**  aplico Prewitt para detectar lineas horinzotales **/
	/** ************************************************* **/
	imgBin = Prewitt(imgBin,0);
	
	/** ************************************************* **/
	/**        Limpio la imagen erosionando 2 veces       **/
	/** ************************************************* **/
	Mat ee = getStructuringElement(MORPH_CROSS,Size(3,3));
	morphologyEx(imgBin,imgBin,MORPH_ERODE,ee);
	ee = getStructuringElement(MORPH_RECT,Size(1,1));
	morphologyEx(imgBin,imgBin,MORPH_ERODE,ee);
	
	/** ************************************************* **/
	/**                      Aplico Hough                 **/
	/** ************************************************* **/
	Mat imgBinH = DibujarHough(imgBin,90);
	ee = getStructuringElement(MORPH_CROSS,Size(4,4));
	morphologyEx(imgBinH,imgBinH,MORPH_ERODE,ee);
	cvtColor(imgBinH,imgBinH,COLOR_BGR2GRAY);
	threshold(imgBinH,imgBinH,0,255,THRESH_BINARY_INV);
	
	/** ******************************************************************* **/
	/** Busco punto superior e inferior de la linea que me define el angulo **/ 
	/** ******************************************************************* **/
	Point2d arriba = LimSuperior(imgBinH);
	cout << " x " << arriba.x << " y " << arriba.y;
	circle(imgBinH,arriba,15,150,2);
	
	Point2d abajo = LimInferior(imgBinH);
	cout << " x " << abajo.x << " y " << abajo.y;
	circle(imgBinH,abajo,15,150,2);
	
	/** ****************************************************** **/
	/** Calculo mediante trigonometria el angulo de la pagina  **/
	/** ****************************************************** **/
	float largo = abajo.y - arriba.y;
	float ancho = abajo.x - arriba.x;
	float angulo = atan(ancho/largo) * 180 / PI;
	
	
	cout <<"\n angulo : " << angulo << " ancho: " << ancho; 
	Mat imgR = rotate(img,-angulo);
	
	
	namedWindow("Imagen",0);
	imshow("Imagen",img);
	namedWindow("ImagenBinH",0);
	imshow("ImagenBinH",imgBinH);
	namedWindow("ImagenRotada",0);
	imshow("ImagenRotada",imgR);
	
	waitKey(0);
	
	return 0;
} 
