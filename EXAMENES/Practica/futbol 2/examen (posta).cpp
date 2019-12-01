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

	Mat img = imread("test2.png"); 
	imshow("Imagen", img);
	waitKey();
	
	Mat nuevaimg;
	cvtColor(img, nuevaimg, COLOR_BGR2GRAY);  // convierto a escala de grises
	
	///Binarizo para remarcar las lineas blancas
	
	threshold(nuevaimg,nuevaimg,254,255,CV_THRESH_BINARY);
	
	imshow("binarizada",nuevaimg);
	waitKey();
	
	
	/// Aplico hough para obtener lineas horizontal y vertical
	
	
	
	
	// Hough
	
	//Canny para detectar los bordes de la imagen
	//cvtColor para cambiar la imagen de escala de grises a BGR y poder mostrar los resultados.
	// nLineas = cantidad minima de lineas a detectar
		
	double nLineas = 130; //160
	
	Mat dst;
	
	Canny(nuevaimg, dst, 254, 255, 3);  //entrada,salida, threshold1, threshold2, apertureSize  (3)
		
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
	
	
	// busco punto centro
	
	for (int i=10;i<cdst.rows;i++){
		
		if ( (x_centro!=0) && (y_centro!=0) ){
			break;
		}
		
		for (int j=2;j<cdst.cols;j++){
			
			if ( ( cdst.at<Vec3b>(i,j)!=negro ) 
			 &&  ( cdst.at<Vec3b>(i+1,j)!=negro ) 
			 &&  ( cdst.at<Vec3b>(i+2,j)!=negro ) 
			 &&  ( cdst.at<Vec3b>(i+3,j)!=negro )
			 &&  ( cdst.at<Vec3b>(i+4,j)!=negro )
			 &&  ( cdst.at<Vec3b>(i+5,j)!=negro )	
           	 &&  ( cdst.at<Vec3b>(i+6,j)!=negro )	
 			 &&  ( cdst.at<Vec3b>(i+7,j)!=negro )
			   )   
			{
				
				x_centro=j;
				y_centro=i;
						
				
				break;
			}
		}
	}
	
	
	
	for (int j=x_centro;j<x_centro+25;j++){
		cdst.at<Vec3b>(y_centro+40,j)=blanco;
		
	}
	
	imshow("line_desde_punto",cdst);
	waitKey();
	
	
	
	Mat image_corta= img.clone();
	Mat primer_corte =image_corta(Rect(0,0,x_centro,image_corta.rows));   //x,y,ancho,alto
	Mat segundo_corte =image_corta(Rect(x_centro+25,0,image_corta.cols-x_centro-25,image_corta.rows));   //x,y,ancho,alto
	
	imshow("roi_primera_parte",primer_corte);
	waitKey();
	
	
	imshow("roi_segunda_parte",segundo_corte);
	waitKey();
	
	
	int pelota=0;
	int conteo1=0;
	
	char color_jugador='n';
	
	
	for (int i=0;i<img.rows;i++){
		
		if (color_jugador!='n'){
			break;
		}
	
		for (int j=0;j<img.cols;j++){
		   
			if (color_jugador!='n'){
				break;
			}
			
			
			if (
				((int)img.at<Vec3b>(i,j)[0]==0)
				&&	((int)img.at<Vec3b>(i,j)[1]==255)
				&&	((int)img.at<Vec3b>(i,j)[2]==255)
				&& ((int)img.at<Vec3b>(i,j+1)[0]==0)
				&&	((int)img.at<Vec3b>(i,j+1)[1]==255)
				&&	((int)img.at<Vec3b>(i,j+1)[2]==255)
				&& ((int)img.at<Vec3b>(i,j+2)[0]==0)
				&&	((int)img.at<Vec3b>(i,j+2)[1]==255)
				&&	((int)img.at<Vec3b>(i,j+2)[2]==255)
				)
			{
				
					for (int q=j;q<j+10;q++){
						
						if (
							((int)img.at<Vec3b>(i-10,q)[0]<80)
							&&	((int)img.at<Vec3b>(i-10,q)[2]>150)
							){
							color_jugador='r';
							break;
						}
							
						if (
								((int)img.at<Vec3b>(i-10,q)[0]>150)
								&&	((int)img.at<Vec3b>(i-10,q)[2]<80)
								){
								color_jugador='a';
								break;
						}	
							
					}
					
					
					
					
					for (int q=j;q<j-10;q--){
						
						if (
							((int)img.at<Vec3b>(i-10,q)[0]<80)
							&&	((int)img.at<Vec3b>(i-10,q)[2]>150)
							){
							color_jugador='r';
							break;
						}
							
							if (
								((int)img.at<Vec3b>(i-10,q)[0]>150)
								&&	((int)img.at<Vec3b>(i-10,q)[2]<80)
								){
								color_jugador='a';
								break;
							}	
								
					}
					
					
				
			}
		}
		
	}
	
	
	
	
	
	
	if (color_jugador=='r'){
		cout<<"pelota del equipo rojo"<<endl;
	}
	
	
	if (color_jugador=='a'){
		cout<<"pelota del equipo azul"<<endl;
	}
	
	
	Vec3b azul(255,0,0);
	
	/*
	//prueba
	
		for (int i=100;i<190;i++){
		for (int j=210;j<320;j++){
			
			img.at<Vec3b>(i,j)=azul;
					}
		}
	
		imshow("azul",img);
		waitKey();
	
		*/
	
	
	
	
	// TRABAJO PARTE IZQUIERDA

	
	Vec3b color_primario=primer_corte.at<Vec3b>(primer_corte.rows/2,5);
	
	
	int pj_x=0;
	int pj_y=0;
	
	
	int cuento_j_izquierda=0;
	int cuento_j_derecha=0;
	
	int cuento_r_izquierda=0;
	int cuento_r_derecha=0;
	
	int cuento_a_izquierda=0;
	int cuento_a_derecha=0;
	
	Mat aux_im = img.clone();
	
	
	for (int i=0;i<primer_corte.rows;i++){
		for (int j=0;j<primer_corte.cols;j++){
			
			if (
				((int)aux_im.at<Vec3b>(i,j)[0]<35)
				&&	((int)aux_im.at<Vec3b>(i,j)[1]<35)
				&&	((int)aux_im.at<Vec3b>(i,j)[2]>210)
				)
			{
				
				cuento_j_izquierda++;
				cuento_r_izquierda++;
				
				for (int q=i-40;q<i+100;q++){
					for (int k=j-30;k<j+120;k++){
						aux_im.at<Vec3b>(q,k)=negro;
					}
				}
				
			}
				
				
				
				
			if (
						((int)aux_im.at<Vec3b>(i,j)[2]<35)
					&&	((int)aux_im.at<Vec3b>(i,j)[1]<70)
					&&	((int)aux_im.at<Vec3b>(i,j)[0]>180)
				)
			{
					
					cuento_j_izquierda++;
					cuento_a_izquierda++;
					
					for (int q=i-40;q<i+100;q++){
						for (int k=j-30;k<j+120;k++){
							aux_im.at<Vec3b>(q,k)=negro;
						}
					}
			}
		}
		
	}
	
	
	
	imshow("aux_img_conteo_izq",aux_im);
	waitKey();
	
	
	
	
	
	
	
	
	
	
	// Segunda
	
	
	
	Mat aux_im2 = img.clone();
	
	
	for (int i=0;i<aux_im2.rows;i++){
		for (int j=primer_corte.cols;j<aux_im2.cols;j++){
			
			if (
				((int)aux_im2.at<Vec3b>(i,j)[0]<35)
				&&	((int)aux_im2.at<Vec3b>(i,j)[1]<35)
				&&	((int)aux_im2.at<Vec3b>(i,j)[2]>210)
				)
			{
				
				cuento_j_derecha++;
				cuento_r_derecha++;
				
				for (int q=i-40;q<i+100;q++){
					for (int k=j-30;k<j+120;k++){
						aux_im2.at<Vec3b>(q,k)=negro;
					}
				}
				
			}
				
				
				
				
				if (
					((int)aux_im2.at<Vec3b>(i,j)[2]<35)
					&&	((int)aux_im2.at<Vec3b>(i,j)[1]<70)
					&&	((int)aux_im2.at<Vec3b>(i,j)[0]>180)
					)
				{
					
					cuento_j_derecha++;
					cuento_a_derecha++;
					
					for (int q=i-40;q<i+100;q++){
						for (int k=j-30;k<j+120;k++){
							aux_im2.at<Vec3b>(q,k)=negro;
						}
					}
				}
		}
		
	}
	
	
	
	imshow("aux_img_conteo_derecha",aux_im2);
	waitKey();
	
	
	
	
	cout<<"cant jugadores izquierda "<<cuento_j_izquierda<<endl;

	cout<<"cant jugadores rojos izquierda "<<cuento_r_izquierda<<endl;
	
	cout<<"cant jugadores azules izquierda "<<cuento_a_izquierda<<endl;
	
	
	
	cout<<"cant jugadores derecha "<<cuento_j_derecha<<endl;
	
	cout<<"cant jugadores rojos derecha "<<cuento_r_derecha<<endl;
	
	cout<<"cant jugadores azules derecha "<<cuento_a_derecha<<endl;
	
	waitKey();
	
	
	
	
	
	
	
	
	
	
	
	
	
	/*
	
	
	imshow("roi_primera_parte_seg",primer_corte);
	waitKey();
	
	
	
	for (int i=0;i<primer_corte.rows;i++){
	for (int j=0;j<primer_corte.cols;j++){
		
		if (
             	((int)primer_corte.at<Vec3b>(i,j)[2]>0)
		){
			
			 primer_corte.at<Vec3b>(i,j)=blanco;
		}
	}
	
	}
	
	
	imshow("roi_primera_parte_seg",primer_corte);
	waitKey();
	
	
	Mat gris1,nueva,thr1;
	
	cvtColor(primer_corte, gris1, COLOR_BGR2GRAY);  //a escala de grises

	threshold(gris1, thr1, 254,255,THRESH_BINARY_INV);  //convierte a binario (menor a 130 lo dejo en 0)
	
	imshow("bin1",thr1);
	waitKey();
	
	
	
	
		/*
	for (int i=0;i<thr1.rows;i++){
		for (int j=0;j<thr1.cols;j++){

			if (thr1.at<Vec3b>(i,j)==blanco){
				
				pj_y=i;
				pj_x=j;
				cuento_j_izquierda++;
				
				
					
				for (int q=i-70;q<i+140;q++){  //160
					for (int k=j-70;k<j+90;k++){  //140
						thr1.at<Vec3b>(q,k)=negro;
					}
				}
				
			}
		}
	}
	
	
	imshow("bin1",thr1);
	waitKey();
	
	
	Mat img_final_izq;
	
	Mat kernel1  = Mat(2,2,0); 
	
	//clavar erode  (limpia imagen)
	erode(thr1, img_final_izq, kernel1, Point(-1, -1), 2, 1, 1);
	
	
	imshow("erode",img_final_izq);
	waitKey();
	
	
	
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img_final_izq, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	
	int d1=contours.size();
	
	
	cout<<"cantidad jugadores izquierda: "<<d1<<endl;
	
	
	
	*/
		
	return 0;
} 










