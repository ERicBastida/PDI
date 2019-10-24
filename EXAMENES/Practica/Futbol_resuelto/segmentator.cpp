#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace pdi;
cv::Mat simhsv,mask,segment,im;
int colorx,colory;
int ancho = 1;
int alto = 1;
int max_thresh = 255;
RNG rng(12345);
void thresh_callback_seg(int, void* );
void thresh_callback_seg_2(int, void* );

void segmentator(cv::Mat src2,cv::Mat &segment2,cv::Mat &mask2,cv::Mat &mask2_inv,int h,int s,int v,int ancho2,int alto2){ //HSV
	//Inputs
	// src: image source
	// h: value of hue
	// s: value of saturation
	// v: value of intensity (it doesn't need intensity value)
	// ancho: H threshold
	// alto: S threshold
	//Outputs
	// segment: image segmented
	// mask: mask of segmented section
	// mask_inv: inverse mask of mask (mask_inv=1-mask)
	
	
	cvtColor(src2,simhsv,CV_BGR2HSV);
	Mat imhsv = simhsv.clone();
	Vec3b color;
	cv::Mat H,S,nueva;
	extractChannel(imhsv,H,0);
	extractChannel(imhsv,S,1);
	color.val[0]=h;//hue
	color.val[1]=s;//hue
	color.val[2]=v;//hue
	//creo la mascara (rectangular)
	int ac_hue,ac_sat,dhue,dsat;
	Vec3b ac_col;
	float val;
	Vec3b uno(1,1,1);Vec3b cero(0,0,0);
	mask2=Mat::zeros(imhsv.size(),CV_8UC3);
	mask2_inv=Mat::zeros(imhsv.size(),CV_8UC3);
	for(int i=0;i<imhsv.rows;i++){
		for(int j=0;j<imhsv.cols;j++){
			ac_hue=(int)H.at<uchar>(i,j);
			ac_sat=(int)S.at<uchar>(i,j);
			dhue=abs(color.val[0]-ac_hue);
			dsat=abs(color.val[1]-ac_sat);
			if(dhue>ancho2/2 or dsat>alto2/2){
				mask2.at<Vec3b>(i,j)=cero;
				mask2_inv.at<Vec3b>(i,j)=uno;}
			else{
				mask2.at<Vec3b>(i,j)=uno;
				mask2_inv.at<Vec3b>(i,j)=cero;}
		}
	}
	segment2=mask2.mul(src2);
}

void segmentator(cv::Mat src2,cv::Mat &segment2,cv::Mat &mask2,cv::Mat &mask2_inv,int cx,int cy,int ancho2,int alto2){ //HSV
	//Inputs
	// src: image source
	// cx: coordinate x of color
	// cy: coordinate y of color
	// ancho: H threshold
	// alto: S threshold
	//Outputs
	// segment: image segmented
	// mask: mask of segmented section
	// mask_inv: inverse mask of mask (mask_inv=1-mask)
	
	
	cvtColor(src2,simhsv,CV_BGR2HSV);
	Mat imhsv = simhsv.clone();
	Vec3b color;
	cv::Mat H,S,nueva;
	extractChannel(imhsv,H,0);
	extractChannel(imhsv,S,1);
	color=imhsv.at<Vec3b>(cy,cx);//pasto (coord invert siempre)
	//creo la mascara (rectangular)
	int ac_hue,ac_sat,dhue,dsat;
	Vec3b ac_col;
	float val;
	Vec3b uno(1,1,1);Vec3b cero(0,0,0);
	mask2=Mat::zeros(imhsv.size(),CV_8UC3);
	mask2_inv=Mat::zeros(imhsv.size(),CV_8UC3);
	for(int i=0;i<imhsv.rows;i++){
		for(int j=0;j<imhsv.cols;j++){
			ac_hue=(int)H.at<uchar>(i,j);
			ac_sat=(int)S.at<uchar>(i,j);
			dhue=abs(color.val[0]-ac_hue);
			dsat=abs(color.val[1]-ac_sat);
			if(dhue>ancho2/2 or dsat>alto2/2){
				mask2.at<Vec3b>(i,j)=cero;
				mask2_inv.at<Vec3b>(i,j)=uno;}
			else{
				mask2.at<Vec3b>(i,j)=uno;
				mask2_inv.at<Vec3b>(i,j)=cero;}
		}
	}
	segment2=mask2.mul(src2);
}

void segmentator(cv::Mat src,cv::Mat &segment,cv::Mat &mask,int cx,int cy){ //HSV
	//Inputs
	// src: image source
	// cx: coordinate x of color
	// cy: coordinate y of color
	// ancho: H threshold
	// alto: S threshold
	//Outputs
	// segment: image segmented
	// mask: mask of segmented section
	
	im=src.clone();//globalizo
	cvtColor(src,simhsv,CV_BGR2HSV);
	colorx=cx;colory=cy;
	/// Create Window
	char* source_window = "Source";
	namedWindow( source_window, CV_WINDOW_AUTOSIZE );
	imshow( source_window, src );
	
	createTrackbar( " H - Ancho:", "Source", &ancho, 180, thresh_callback_seg );
	createTrackbar( " S - Alto:", "Source", &alto, 255, thresh_callback_seg_2 );
	thresh_callback_seg( 0, 0 );
	thresh_callback_seg_2( 0, 0 );
	
	waitKey(0);
	
}
void thresh_callback_seg_2(int, void* )
{
	Mat imhsv = simhsv.clone();
	Vec3b color;
	cv::Mat H,S,nueva;
	extractChannel(imhsv,H,0);
	extractChannel(imhsv,S,1);
	color=imhsv.at<Vec3b>(colory,colorx);//pasto (coord invert siempre)
	//creo la mascara (rectangular)
	int ac_hue,ac_sat,dhue,dsat;
	Vec3b ac_col;
	float val;
	Vec3b uno(1,1,1);Vec3b cero(0,0,0);
	mask=Mat::zeros(imhsv.size(),CV_8UC3);
	for(int i=0;i<imhsv.rows;i++){
		for(int j=0;j<imhsv.cols;j++){
			ac_hue=(int)H.at<uchar>(i,j);
			ac_sat=(int)S.at<uchar>(i,j);
			dhue=abs(color.val[0]-ac_hue);
			dsat=abs(color.val[1]-ac_sat);
			if(dhue>ancho/2 or dsat>alto/2)
				mask.at<Vec3b>(i,j)=cero;
			else
				mask.at<Vec3b>(i,j)=uno;
		}
	}
	segment=mask.mul(im);
	
	namedWindow("H");
	namedWindow("S");
	namedWindow("nueva");
	imshow("H",H);
	imshow("S",S);
	imshow("nueva",segment);
}
void thresh_callback_seg(int, void* )
{
	Mat imhsv = simhsv.clone();
	Vec3b color;
	cv::Mat H,S,nueva;
	extractChannel(imhsv,H,0);
	extractChannel(imhsv,S,1);
	color=imhsv.at<Vec3b>(colory,colorx);//pasto (coord invert siempre)
	//creo la mascara (rectangular)
	int ac_hue,ac_sat,dhue,dsat;
	Vec3b ac_col;
	float val;
	Vec3b uno(1,1,1);Vec3b cero(0,0,0);
	mask=Mat::zeros(imhsv.size(),CV_8UC3);
	for(int i=0;i<imhsv.rows;i++){
		for(int j=0;j<imhsv.cols;j++){
			ac_hue=(int)H.at<uchar>(i,j);
			ac_sat=(int)S.at<uchar>(i,j);
			dhue=abs(color.val[0]-ac_hue);
			dsat=abs(color.val[1]-ac_sat);
			if(2*dhue>ancho or 2*dsat>alto)
				mask.at<Vec3b>(i,j)=cero;
			else
				mask.at<Vec3b>(i,j)=uno;
		}
	}
	segment=mask.mul(im);
	
	namedWindow("H");
	namedWindow("S");
	namedWindow("nueva");
	imshow("H",H);
	imshow("S",S);
	imshow("nueva",segment);
}
