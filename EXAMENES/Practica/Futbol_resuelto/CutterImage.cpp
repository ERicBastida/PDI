#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


void CutterImage(cv::Mat im,cv::Mat &out){ //TESTEAR BIEN
	//Input
	// im: image with white background
	//Output
	// out: output image cutted
	
	int left_max=im.cols,right_max=0,top_max=im.rows,bottom_max=0;
	cv::Mat orig=im.clone();
	if(im.channels()==3){
		cvtColor(im, im, CV_BGR2GRAY);
	}
		threshold(im,im,254,1,THRESH_BINARY_INV);
	//left border
	for(int i=0;i<im.rows;i++)
		for(int j=0;j<im.cols;j++){
		if(im.at<uchar>(i,j)==1){
			if(j<left_max){
				left_max=j;
			}
		}
	}
	//right border
	for(int i=0;i<im.rows;i++)
		for(int j=im.cols-1;j>=0;j--){
		if(im.at<uchar>(i,j)==1){
			if(j>right_max){
				right_max=j;
			}
		}
	}
	//top border
	for(int j=0;j<im.cols;j++){
		for(int i=0;i<im.rows;i++)
			if(im.at<uchar>(i,j)==1){
			if(i<top_max){
				top_max=i;
			}
		}
	}
	//bottom border
	for(int j=0;j<im.cols;j++){
		for(int i=im.rows-1;i>=0;i--)
			if(im.at<uchar>(i,j)==1){
			if(i>bottom_max){
				bottom_max=i;
			}
		}
	}
	//RoI
	out=orig( Rect(left_max,top_max,right_max-left_max+1, bottom_max-top_max+1) );
	return;
}
