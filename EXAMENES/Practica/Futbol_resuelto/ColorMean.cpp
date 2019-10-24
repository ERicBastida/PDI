#include<iostream>
#include "pdi_functions.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <random>
using namespace cv;
int cant_click_color_mean=0, x_click_color_mean, y_click_color_mean;
cv::Scalar col_click_color_mean;
void CallBackFunc_color_mean(int event, int x, int y, int flags, void* userdata)
{	
	if  ( event == EVENT_LBUTTONDOWN )
	{
		if(cant_click_color_mean==1){
			Mat* rgb = (Mat*) userdata;
			cv::Mat roi=(*rgb)( Rect(x_click_color_mean,y_click_color_mean,x-x_click_color_mean+1, y-y_click_color_mean+1) );
			cv::Scalar col_click_color_mean = cv::mean(roi);
			std::cout<<col_click_color_mean;
			return;
		}
		x_click_color_mean=x;
		y_click_color_mean=y;
		cant_click_color_mean++;
	}
	
}

void ColorMean(cv::Mat im, cv::Scalar &color){ 
	//Input
	// im: image
	// mouse: select 2 points in corresponding order (upper and lower)
	
	//Output
	// color: Scalar with color mean of color in the rectangle selected
	
	
	if ( im.empty() ) 
	{ 
		std::cout << "Error loading the image" << std::endl;
		return; 
	}
	namedWindow("Image", 1);
	setMouseCallback("Image", CallBackFunc_color_mean, &im);
	imshow("Image", im);
	color=col_click_color_mean;
	waitKey();
	return;
}
