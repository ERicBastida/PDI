#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;

void FindColChanges(cv::Mat im, int row,int tol,std::vector<int> &coord,int &cant){
	// This function find changes in intensity of a line in some grayscale image.
	//Inputs:
	//im: image input
	//row: number of row for scanning
	//tol: tolerance for detect a change
	//Output:
	//coord: vector of coordinates of intensity changes
	//cant: total of intensity changes
	cant=0;
	Scalar ante,post;
	for(int j=0;j<im.cols-1;j++){
		ante=im.at<uchar>(row,j);
		post=im.at<uchar>(row,j+1);
		if(abs(ante.val[0]-post.val[0])>tol){
			coord.push_back(j);
			cant++;
		}
	}
}
