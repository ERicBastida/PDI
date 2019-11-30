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



int main(int argc, char** argv) {
	
	 
	/// cargo imagen original

	Mat img = imread("Avion8.png"); 
	imshow("Imagen", img);
	waitKey();
	
	Mat nuevaimg;
	cvtColor(img, nuevaimg, COLOR_BGR2GRAY);  // convierto a escala de grises
	
	///Binarizo para remarcar las lineas blancas
	
	threshold(nuevaimg,nuevaimg,200,255,CV_THRESH_BINARY);
	
	imshow("binarizada",nuevaimg);
	waitKey();
	
	
	/// Aplico hough para obtener lineas horizontal y vertical
	
	
	
	
	// Hough
	
	//Canny para detectar los bordes de la imagen
	//cvtColor para cambiar la imagen de escala de grises a BGR y poder mostrar los resultados.
	// nLineas = cantidad minima de lineas a detectar
		
	double nLineas = 130; //160
	
	Mat dst;
	
	Canny(nuevaimg, dst, 220, 255, 3);  //entrada,salida, threshold1, threshold2, apertureSize  (3)
		
	double	ro = 1,
	tita = CV_PI/180; // = 1
		
	// NO TOCAR ESOS VALORES DE ARRIBA!!!!
		
	// ro grado a donde va desde 0 a 1 pixel a pixel
	// tita colinealidad (dis) grado a grado
		
	vector<Vec2f> lines;
	HoughLines(dst, lines,ro,tita, nLineas, 0, 0 );   // ro y tita no hacen nada, no tocarlos
		
	//busca de mas larga a mas corta
	//en lineas arranca desde el 0,0 y te las ordena de mas grande a mas chica desde lo mas a la izquierda y mas arriba posible
		
	//theta es el angulo de la perpendicular (restarle 90 para saber cual es, y para usarlo despues sumarle 90)
	//rho te indica la distancia PERPENDICULAR desde el origen hasta el primer punto que encuentre
	
	imshow("hough_principio",dst);
	waitKey();	
	
	//este hough es el comun te devuelve rho y theta, buscar el houghp (de probabilidad) que te devuelve el punto desde donde arranca cada linea que encuentra
	
	//Mat cdst(img.rows, img.cols, CV_8UC3, Scalar(0,0,0));
	
	Mat cdst(img.rows, img.cols, CV_8UC(3), Scalar(0,0,0));
	
	for ( size_t i = 0; i < 2; i++ ) {   // podria ir hasta lines.size()
		
		float rho = lines[i][0], theta = lines[i][1];    // aca te devuelve rho y theta de mayor a menor lineas
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		//line( cdst, pt1, pt2, Scalar(0,0,255), 1, CV_AA);
		line( cdst, pt1, pt2, 255, 1);
		
	}
		

	imshow("Lineas_Hough",cdst);
	waitKey();
	
	
	// busco punto centro entre linea horizontal y lateral
	
	
	Vec3b negro(0,0,0);
	
	Vec3b rojo(0,0,255);
	
	Vec3b blanco(255,255,255);
	
	int x_centro=0;
	int y_centro=0;
	
	
	
	
	Mat aux_img=img.clone();
	Mat aux_img2=img.clone();
	
	
	
	// segmento
	
	
	for (int i=0;i<aux_img.rows;i++){
		for (int j=0;j<aux_img.cols;j++){
			
			if (
				((int)aux_img.at<Vec3b>(i,j)[2]<160)
			 || ((int)aux_img.at<Vec3b>(i,j)[1]>40)
		     || ((int)aux_img.at<Vec3b>(i,j)[0]>40)
				){
				aux_img.at<Vec3b>(i,j)=negro;
			}
		}
	}
	
	
	
	imshow("Seg",aux_img);
	waitKey();
	
	
	
	for (int i=0;i<aux_img2.rows;i++){
		for (int j=0;j<aux_img2.cols;j++){
			
			if (
				((int)aux_img2.at<Vec3b>(i,j)[2]<160)
				|| ((int)aux_img2.at<Vec3b>(i,j)[1]<160)
				|| ((int)aux_img2.at<Vec3b>(i,j)[0]<160)
				){
				aux_img2.at<Vec3b>(i,j)=negro;
			}
		}
	}
	
	
	imshow("Seg2",aux_img2);
	waitKey();
	

	
	
	char lado_avion='n';
	
	int posx=0;
	
	for (int i=0;i<aux_img2.rows;i++){
		
		if (lado_avion!='n')
			break;
		
			if (aux_img2.at<Vec3b>(i,2)!=negro){
				lado_avion='d';
				break;
			}
	}
	
	
	for (int i=0;i<aux_img2.rows;i++){
		
		if (lado_avion!='n')
			break;
		
		if (aux_img2.at<Vec3b>(i,aux_img2.cols-2)!=negro){
			lado_avion='i';
			break;
		}
	}
	
	
	
	
	for (int j=0;j<aux_img2.cols;j++){
		
		if (posx!=0)
			break;
		
		if (aux_img2.at<Vec3b>(2,j)!=negro){
			posx=j;
			break;
		}
	}
	
	
	for (int j=0;j<aux_img2.cols;j++){
		
		if (posx!=0)
			break;
		
		if (aux_img2.at<Vec3b>(aux_img2.rows-2,j)!=negro){
			posx=j;
			break;
		}
	}
	
	
	
	if (posx!=0){
		if (posx>aux_img2.cols/2){
			lado_avion='i';
		}
		else{
			lado_avion='d';
		}
	}
	
	cout<<"lado de avion: "<<lado_avion<<endl;;
	
	
	int colision=0;
	
	
	for (int i=0;i<aux_img.rows;i++){
		for (int j=0;j<aux_img.cols;j++){
			
			if ( (aux_img.at<Vec3b>(i,j)!=negro) && (cdst.at<Vec3b>(i,j)!=negro) ){
				colision=1;
			}
		}
	}
	
	
	
	
	
	
	double grados = (lines[0][1]*180/CV_PI);
	
	if (colision==1){
	cout<<"el objeto colisiona"<<endl;
	}
	else{
		cout<<"el objeto no colisiona"<<endl;
	}
	
	
	
	grados=round(grados);
	
	
		/*
	if (lado_avion=='i'){
		grados=round(grados+180);
	}
	*/
	
	// FALTA DETECTAR DIRECCION DE AVION Y CALCULO DE GRADOS
	
	
	
	// primer cuadrante 1 a 45 en HoughLines
	
	
	/*
	if (grados==90){
		grados=round(grados-90);
	}
	else if (grados==0){
		grados=round(grados+90);
	}
	
	
	
	if (grados>90){
	grados=round(grados+180+45);
	}
	if (grados<90){
		grados=round(grados+90+45);
	}
	*/
	
	cout<<"los grados de trayectoria son aproximadamente: "<<grados<<endl;
	
		
	return 0;
} 










