#include<iostream>
#include "pdi_functions.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <random>
using namespace cv;
void CallBackFunc_coord(int event, int x, int y, int flags, void* userdata)
{	

	if  ( event == EVENT_LBUTTONDOWN )
	{
		std::cout << "Left button is clicked - position (" << x << ", " << y << ")" << std::endl;
	}
	else if  ( event == EVENT_RBUTTONDOWN )
	{
		std::cout << "Right button is clicked - position (" << x << ", " << y << ")" << std::endl;
	}
	else if  ( event == EVENT_MBUTTONDOWN )
	{
		std::cout << "Middle button is clicked - position (" << x << ", " << y << ")" << std::endl;
	}
	else if ( event == EVENT_MOUSEMOVE )
	{
		std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;
	}  
	Mat* rgb = (Mat*) userdata;

	if ((int)(*rgb).channels()==3) //RGB 
	{ 
		cv::Vec3b col=(*rgb).at<cv::Vec3b>(y, x);
		cv::Mat M(1,1, CV_8UC3, cv::Scalar(0,0,0));
		M.at<cv::Vec3b>(0,0)=col;
		std::printf("Color RGB: %d, %d, %d\n", 
			   (int)(*rgb).at<Vec3b>(y, x)[0], 
			   (int)(*rgb).at<Vec3b>(y, x)[1], 
			   (int)(*rgb).at<Vec3b>(y, x)[2]); 
		cvtColor(M,M,CV_BGR2HSV);
		std::printf("Color HSV: %d, %d, %d\n", 
					(int)(M).at<Vec3b>(0, 0)[0], 
					(int)(M).at<Vec3b>(0, 0)[1], 
					(int)(M).at<Vec3b>(0, 0)[2]); 
		
	}
	else
	{
		printf("Color: %d, %d, %d\n", 
			    (int)(*rgb).at<uchar>(y, x));
	}

}

void view_coordinates(cv::Mat im){ 
	if ( im.empty() ) 
	{ 
		std::cout << "Error loading the image" << std::endl;
		return; 
	}
	namedWindow("Image", 1);
	setMouseCallback("Image", CallBackFunc_coord, &im);
	imshow("Image", im);
	waitKey();
}
