#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

int computeOutput(int, int, int, int, int);
//Input:                          //TERMINAR
// image: image in gray scale
// r1: coordinate x of first point
// s1: coordinate y of first point
// r2: coordinate x of second point
// s2: coordinate y of second point
//Output:
// new_image: image with new contrast

void ContrastStretching(cv::Mat image, cv::Mat &new_image, int r1, int s1, int r2, int s2){
	new_image = image.clone();

	for(int y = 0; y < image.rows; y++){
		for(int x = 0; x < image.cols; x++){
				int output = computeOutput(image.at<uchar>(y,x), r1, s1, r2, s2);
				new_image.at<uchar>(y,x)= saturate_cast<uchar>(output);
		}
	}
	return;
}

int computeOutput(int x, int r1, int s1, int r2, int s2){
	float result;
	if(0 <= x && x <= r1){
		result = s1/r1 * x;
	}else if(r1 < x && x <= r2){
		result = ((s2 - s1)/(r2 - r1)) * (x - r1) + s1;
	}else if(r2 < x && x <= 255){
		result = ((255 - s2)/(255 - r2)) * (x - r2) + s2;
	}
	return (int)result;
}
