#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


void Skeleton(cv::Mat img,cv::Mat &skel){
	//Input:
	// img: image in gray scale
	//Output:
	// skel: skeleton of img
	
	cv::threshold(img, img, 127, 255, cv::THRESH_BINARY); 
	cv::Mat skel2(img.size(), CV_8UC1, cv::Scalar(0));
	skel=skel2.clone();
	cv::Mat temp;
	cv::Mat eroded;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;		
	do
	{
		cv::erode(img, eroded, element);
		cv::dilate(eroded, temp, element); // temp = open(img)
		cv::subtract(img, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(img);
		
		done = (cv::countNonZero(img) == 0);
	} while (!done);
	return;
}
