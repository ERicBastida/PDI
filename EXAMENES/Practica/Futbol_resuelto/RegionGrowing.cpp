#include<iostream>
#include "pdi_functions.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <random>
using namespace cv;

void RegionGrowing(cv::Mat im, cv::Mat &result, cv::Mat &mask, int x, int y, double lodiff, double updiff, int neighbor=4){
	//Inputs
	// im: Image
	// x: Coordinate x of seed
	// y: Coordinate y of seed
	// lodiff: Low threshold of growing
	// updiff: Up threshold of growing
	// neighbor: Size of neighbor (4 or 8)
	
	//Output
	// result: Original image painted with in the region
	// mask: Mask of new region, if you wanna inverse make mask=1-mask;
	
	cv::Mat image;
	mask=Mat::zeros(im.size(),im.type());
	im.convertTo(image,CV_32F);
	float k;
	result=im.clone();
	floodFill(image,Point(x,y),256,NULL,lodiff,updiff,neighbor); 
	for(int i=0;i<image.rows;i++)
		for(int j=0;j<image.cols;j++){
		k=image.at<float>(i,j);
		if(k==256){
			result.at<uchar>(i,j)=255; // 255 is the new value
			mask.at<uchar>(i,j)=1;
		}
	}
	imshow("result",result);
}
