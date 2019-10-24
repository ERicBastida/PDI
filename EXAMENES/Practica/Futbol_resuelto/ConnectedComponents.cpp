#include<iostream>
#include "pdi_functions.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <random>
using namespace cv;

void ConnectedComponents (cv::Mat &imagen ){
	//Input
	// imagen: image with black background.
	//SI EL FONDO DE LA IMAGEN ES BLANCO DESCOMENTAR LA LINEA SIGUIENTE:
	//threshold(src,src,252,255,THRESH_TOZERO_INV); //mover el 252 a un valor optimo
	
	
	//Find contours in a binary image
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(imagen, contours, hierarchy , CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );
	
	//Draw contours
	RNG rng(12345);
	Mat drawing = Mat::zeros( imagen.size(), CV_8UC3 );
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );
	
	for( int i = 0; i < contours.size(); i++ )
	{ 
		approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
		minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
	}
	
	
	Mat img2 = imagen.clone();
	
	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
		
		circle( img2, center[i], (int)radius[i], color, 2, 8, 0 );
		
		//escribo numero de componente
		string box_text = format("%d", i);
		// calcular la posicion para el texto anotado:
		int pos_x = center[i].x;
		int pos_y = center[i].y;
		// y ahora poner en la imagen:
		putText(drawing, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, .8, color, 1.0);
	}
	
	namedWindow("Componentes conectadas" , 1 );
	imshow("Componentes conectadas", drawing);
	waitKey(1);
	
	vector<double> areas(contours.size() );
	
	for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
	{
		areas[i] = contourArea( contours[i], false);
		std::cout<<"Area comp "<<i<<" : "<<areas[i]<<std::endl;
	}
	
	imshow("Imagen", img2);
	
	std::cout<<"Cantidad de Componentes: "<<contours.size()<<std::endl;
}
