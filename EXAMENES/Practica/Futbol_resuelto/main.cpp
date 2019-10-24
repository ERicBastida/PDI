#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"

using namespace cv;
using namespace pdi;
void view_coordinates(cv::Mat im);
void segmentator(cv::Mat src,cv::Mat &segment,cv::Mat &mask,int cx,int cy);
void segmentator(cv::Mat src2,cv::Mat &segment2,cv::Mat &mask2,cv::Mat &mask2_inv,int h,int s,int v,int ancho2,int alto2);
void Huang(cv::Mat im,cv::Mat &cdst,cv::Mat &dst,double threshold,int &cx,int &cy,bool vertical=0,bool horizontal=0);
void ColorMean(cv::Mat im, cv::Scalar &color);
int main(int argc, char** argv) {
	//read image
	cv::Mat im=imread("Fut02_3.png"),output,mask,segment,mask_inv,im_hsv,lines,lines2,jugadores,mcolor,jugadores_orig,jugadores_sinc,jugadores_grande,salida;
	//Fut01_1,Fut01_3,Fut02_1
	cv::Mat fich=imread("fich.jpg"),unl=imread("unl.jpg"),sinc=imread("Logo03.png");
	int cx=0,cy=0,cy2;
	//view_coordinates(im);//261 225
	//segmentator(im,segment,mask,96,132);//15 87
	int h1,s1,anc1,al1;
	//HOUGH
	Huang(im,lines,lines2,220,cx,cy,1,0); //vertical
	//std::cout<<cy<<std::endl;
	Huang(im,lines,lines2,500,cx,cy2,0,1);//horizomntal
	//std::cout<<cx;
	//SEGMENTO
		Scalar color;
		mcolor = im( Rect(cy+10,cx+120,unl.cols,unl.rows));
		cvtColor(im,im_hsv,CV_BGR2HSV);
		extractChannel(im_hsv,mcolor,0);
		color=cv::mean(mcolor);
	std::cout<<color.val[0];
	if(color.val[0]>40 and color.val[0]<55){
		h1=38;
		s1=125;
		anc1=15;
		al1=87;
	}
	else{
		h1=18;
		s1=98;
		anc1=10;
		al1=87;	
		}
	
	//view_coordinates(im);
	//segmentator(im,segment,mask,354,233);
	segmentator(im,segment,mask,mask_inv,h1,s1,1,anc1,al1);
	jugadores=im-segment;
	jugadores_orig=jugadores;
	//HAGO IMAGENES PEQUEÑAS
	unl.copyTo(im(cv::Rect(cy,cx,unl.cols, unl.rows)));//pego unl (sacar de aca el 10 y 120) 
	fich.copyTo(im(cv::Rect(cy-fich.cols-3,cx,fich.cols, fich.rows)));//pego fich
	sinc.copyTo(im(cv::Rect(cy-sinc.cols/2,(im.rows-sinc.rows)-2,sinc.cols, sinc.rows)));//pego sinc
	//TODOS LOS JUGADORES DEL CARTEL FICH Y UNL
	jugadores = jugadores( Rect(cy-fich.cols-3,cx-10,(fich.cols*2)+15,fich.rows+15));//corto jugadores al cartel
	jugadores_grande=im*0;
	jugadores.copyTo(jugadores_grande(cv::Rect(cy-fich.cols-3,cx-10,jugadores.cols,jugadores.rows)));
	//JUGADORES DEL SINC
	jugadores_orig = jugadores_orig( Rect(cy-sinc.cols/2,(im.rows-sinc.rows)-2,sinc.cols,sinc.rows));//corto jugadores al cartel
	jugadores_sinc=im*0;
	jugadores_orig.copyTo(jugadores_sinc(cv::Rect(cy-sinc.cols/2,(im.rows-sinc.rows)-2,jugadores_orig.cols,jugadores_orig.rows)));
	
	//	salida=jugadores;
	namedWindow("Output",1);
	//imshow("Output", mcolor);
	
	erode(jugadores_sinc,jugadores_sinc,getStructuringElement(MORPH_CROSS,Size(3,3)));
	dilate(jugadores_sinc,jugadores_sinc,getStructuringElement(MORPH_CROSS,Size(3,3)));
	imshow("Output", jugadores_sinc+im+jugadores_grande*0.75);
	//wait for the user to press any key:
	waitKey(0);
	return 0;
} 

