#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


void CutterMask(cv::Mat mask,cv::Mat &mask_cutted,cv::Mat &im){ //TESTEAR BIEN
	//Input
	// mask: mask with 1 or 3 channels
	//Output
	// mask_cutted: output mask cutted
	// im: image cutted with size of mask
	
	int left_max=mask.cols,right_max=0,top_max=mask.rows,bottom_max=0;
	if(mask.channels()==3){
	cvtColor(mask, mask, CV_BGR2GRAY);
	threshold(mask,mask,0,1,THRESH_BINARY);
	}
	
	//left border
	for(int i=0;i<mask.rows;i++)
		for(int j=0;j<mask.cols;j++){
			if(mask.at<uchar>(i,j)==1){
				if(j<left_max){
					left_max=j;
				}
			}
	}
	//right border
	for(int i=0;i<mask.rows;i++)
		for(int j=mask.cols-1;j>=0;j--){
			if(mask.at<uchar>(i,j)==1){
				if(j>right_max){
					right_max=j;
				}
			}
	}
	//top border
	for(int j=0;j<mask.cols;j++){
		for(int i=0;i<mask.rows;i++)
		if(mask.at<uchar>(i,j)==1){
			if(i<top_max){
				top_max=i;
			}
		}
	}
	//bottom border
	for(int j=0;j<mask.cols;j++){
		for(int i=mask.rows-1;i>=0;i--)
			if(mask.at<uchar>(i,j)==1){
			if(i>bottom_max){
				bottom_max=i;
			}
		}
	}
	//RoI
	cvtColor(mask, mask, CV_GRAY2BGR);
	mask_cutted=mask( Rect(left_max,top_max,right_max-left_max,bottom_max-top_max) );
	im=im( Rect(left_max,top_max,right_max-left_max+1,bottom_max-top_max+1) );
	return;
}
