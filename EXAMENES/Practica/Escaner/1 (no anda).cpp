#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"
#include <iostream>
#include <opencv2/core/types_c.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <ostream>
#include "opencv2/core/core.hpp"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>




using namespace std;
using namespace cv;
using namespace pdi;














Mat DibujarHough(Mat img, double nLineas){
	//Canny para detectar los bordes de la imagen
	//cvtColor para cambiar la imagen de escala de grises a BGR y poder mostrar los resultados.
	// nLineas = cantidad minima de lineas a detectar
	Mat dst, cdst;
	Canny(img, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	
	
	double	tita = 1,
		ro = CV_PI/4; // = 1
	//			nlineas = 50; //cantidad minima de lineas a detectar
	
	vector<Vec2f> lines;
	HoughLines(dst, lines,tita,ro, nLineas, 0, 0 );
	
	for ( size_t i = 0; i < lines.size(); i++ ) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
	}
	
	return cdst;
}




Mat DetectarLineas(Mat img, int tipo){
	
	//Tipo: 0 = Vertical
	//		1 = Horizontal
	//		2 = Diagnoal /
	//		3 = Diagonal \
	
	Mat Prewitt1;
	Mat kernel = Mat::eye( 3, 3, CV_32FC1 );
	switch(tipo){
	case 0 : //Vertical
		
		*(((float*) (kernel.data))) = -1;  // [0][0] 
		*(((float*) (kernel.data))+1) = 2;  // [0][1] 
		*(((float*) (kernel.data))+2) = -1.0;  // [0][2] 
		*(((float*) (kernel.data))+3) = -1.0; //[1][0]
		*(((float*) (kernel.data))+4) = 2.0;// [1][1]
		*(((float*) (kernel.data))+5) = -1.0;// [1][2]
		*(((float*) (kernel.data))+6) = -1.0;// [2][0]
		*(((float*) (kernel.data))+7) = 2.0;// [2][1]
		*(((float*) (kernel.data))+8) = -1.0;// [2][2]
		
		filter2D(img, Prewitt1, -1, kernel);
		break;
	case 1 : //Horizontal
		
		
		*(((float*) (kernel.data))) = -1.0;  // [0][0] 
		*(((float*) (kernel.data))+1) = -1.0;  // [0][1] 
		*(((float*) (kernel.data))+2) = -1.0;  // [0][2] 
		*(((float*) (kernel.data))+3) = 2.0; //[1][0]
		*(((float*) (kernel.data))+4) = 2.0;// [1][1]
		*(((float*) (kernel.data))+5) = 2.0;// [1][2]
		*(((float*) (kernel.data))+6) = -1.0;// [2][0]
		*(((float*) (kernel.data))+7) = -1.0;// [2][1]
		*(((float*) (kernel.data))+8) = -1.0;// [2][2]
		
		filter2D(img, Prewitt1, -1, kernel);
		return Prewitt1;
	case 2 : //Diagonal /
		
		
		*(((float*) (kernel.data))) = -1.0;  // [0][0] 
		*(((float*) (kernel.data))+1) = -1.0;  // [0][1] 
		*(((float*) (kernel.data))+2) = 2.0;  // [0][2] 
		*(((float*) (kernel.data))+3) = -1.0; //[1][0]
		*(((float*) (kernel.data))+4) = 2.0;// [1][1]
		*(((float*) (kernel.data))+5) = -1.0;// [1][2]
		*(((float*) (kernel.data))+6) = 2.0;// [2][0]
		*(((float*) (kernel.data))+7) = -1.0;// [2][1]
		*(((float*) (kernel.data))+8) = -1.0;// [2][2]
		
		filter2D(img, Prewitt1, -1, kernel);
		return Prewitt1;
	case 3 : //Diagonal \
		
		
		*(((float*) (kernel.data))) = 2.0;  // [0][0] 
		*(((float*) (kernel.data))+1) = -1.0;  // [0][1] 
		*(((float*) (kernel.data))+2) = -1.0;  // [0][2] 
		*(((float*) (kernel.data))+3) = -1.0; //[1][0]
		*(((float*) (kernel.data))+4) = 2.0;// [1][1]
		*(((float*) (kernel.data))+5) = -1.0;// [1][2]
		*(((float*) (kernel.data))+6) = -1.0;// [2][0]
		*(((float*) (kernel.data))+7) = -1.0;// [2][1]
		*(((float*) (kernel.data))+8) = 2.0;// [2][2]
		
		filter2D(img, Prewitt1, -1, kernel);
		return Prewitt1;
	}
	
	
}





Mat Segmentar(Mat imagen, int Hmin, int Hmax,int Smin, int Smax){
	//Variables
	Mat mascara = cv::Mat::zeros(imagen.size(),imagen.type());
	vector<Mat> hsv_channels(3);
	Mat imgHSV;
	
	//Paso de RGB a HSI
	cvtColor(imagen,imgHSV,COLOR_BGR2HSV);
	//Divido los canales
	split(imgHSV,hsv_channels);
	
	
	///Recorro las componentes H y S, entonces si estos valores están dentro de 
	///los minimos y maximos definidos anteriormente a la mascara le coloco 1,
	///sino queda en cero. Entonces en otras palabras, si encuentra el amarillo
	/// a la mascara la dejo en 1, de lo contrario 0.
	for (int i = 0; i<imagen.rows;i++)
		for (int j = 0; j<imagen.cols;j++){
			int H = (int)hsv_channels[0].at<uchar>(i,j);
			int S = (int)hsv_channels[1].at<uchar>(i,j);
			if ((Hmin<H)&&(H<Hmax) && (Smin<S)&&(S<Smax)){
				mascara.at<cv::Vec3b>(i,j)[0]=1; //Blue
				mascara.at<cv::Vec3b>(i,j)[1]=1; //Green
				mascara.at<cv::Vec3b>(i,j)[2]=1; //Red
			}
	}
		
		///Multiplico la imagen con la mascara y asi obtengo solo el cartel
		Mat imgSegmentada;
		multiply(imagen,mascara,imgSegmentada); //Multiplico la imagen por la mascara
		
		return imgSegmentada;
}














//Recorro horizontalmente
Point2d PrimerPunto(Mat imagen){
	Point2d p;
	for(int j=0; j< (int)imagen.cols/2; j++){
		for(int i=0; i< imagen.rows; i++){
			int r = (int)imagen.at<uchar>(i,j);
			if(r != 0){
				p.x = j;
				p.y = i;
				return p;
			}
		}
	}
}



Point2d SegundoPunto(Mat imagen){
	Point2d p;
	for(int j= (int) imagen.cols-1; j>(int) imagen.cols/2; j--){
		for(int i=0; i< imagen.rows; i++){
			int r = (int)imagen.at<uchar>(i,j);
			if(r != 0){
				p.x = j;
				p.y = i;
				return p;
			}
		}
	}
}



float ObtenerAngulo(Mat imagen){
	float angulo = 0;
	Point2d p1,p2,vertice;
	
	p1 = PrimerPunto(imagen);
	p2 = SegundoPunto(imagen);
	
	//Si el angulo de rotacion es negativo
	if(p1.y < p2.y){
		Point2d aux;
		aux = p1;
		p1 = p2;
		p2 = aux;
	}
	
	vertice.x = p1.x;
	vertice.y = p2.y;
	
	float CO = abs(p1.y-vertice.y);
	float CA = abs(p2.x-vertice.x);
	
	//Para ver el angulo de rotacion
	Mat img;
	img = imagen.clone();
	line(img,p1,p2,255,4);
	line(img,p2,vertice,255,4);
	line(img,p1,vertice,255,4);
	namedWindow("triangulo",0);
	imshow("triangulo",img);
	
	angulo = atan(CO/CA) * 180 / CV_PI;
	return angulo;
	
}


///NO ESTA ANDANDO DEL TODO BIEN, PERO EL ANGNGULO SI ESTA BIEN,
///TENGO QUE ACOMODAR LAS LINEAS DE HOUGH, TRATAR DE QUE QUEDE SOLO UNA



int main(int argc, char** argv) {
	Mat imagen = imread("escaneo3.jpg",IMREAD_GRAYSCALE);
	Mat Img = imagen.clone();
	
	namedWindow("Imagen",0);
	imshow("Imagen",imagen);
	
	///Binarizo la imagen
	threshold(imagen,imagen,200,255,THRESH_BINARY);
	//	namedWindow("Binarizada",0);
	//	imshow("Binarizada",imagen);
	
	
	///Detecto las Lineas en todas las direcciones
	Mat imagen0 = DetectarLineas(imagen,0);
	Mat imagen1 = DetectarLineas(imagen,1);
	Mat imagen2 = DetectarLineas(imagen,2);
	Mat imagen3 = DetectarLineas(imagen,3);
	imagen = imagen0+imagen1+imagen2+imagen3;
	//	namedWindow("Lineas",0);
	//	imshow("Lineas",imagen);
	
	
	///Aplico Hough
	double nLineas = 340;
	Mat imagenHough = DibujarHough(imagen,nLineas);
	namedWindow("Hough",0);
	imshow("Hough",imagenHough);
	
	
	///Obtengo el angulo de rotacion
	float angulo = ObtenerAngulo(imagenHough);
	cout<< angulo;
	
	
	Img = rotate(Img,angulo);
	namedWindow("Rotada",0);
	imshow("Rotada",Img);
	
	waitKey(0);
	return 0;
} 

