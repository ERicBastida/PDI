#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

void Huang(cv::Mat im,cv::Mat &cdst,cv::Mat &dst,double threshold,int &cx,int &cy,bool vertical=0,bool horizontal=0){
	//Inputs:
	// YOU CAN MOVE the Canny threshold also
	// Im: image
	// threshold: for HuangLines
	// vertical and horizontal: for detect vertical or horizontal lines
	//Outputs:
	// cdst: image and HoughLines
	// dst: contour of image
	//Notes: you can modify the degrees at line 30 and 31, also you can modify the large in 38 to 41

	
	Canny(im, dst, 50, 200, 3); //MOVE THIS PARAMETERS IF YOU NEED
	cvtColor(dst, cdst, CV_GRAY2BGR); 
	cx=0;cy=0;
	vector<Vec2f> lines;
	// detect lines
		HoughLines(dst, lines, 1, CV_PI/180, threshold, 0, 0 );
		int ymin=0;
	// draw lines
		for( size_t i = 0; i < lines.size(); i++ ){
		float rho = lines[i][0], theta = lines[i][1];
		// tener en cuenta sistema x-y normal y 0<theta<180
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		if(( theta>(CV_PI/180)*170 || theta<(CV_PI/180)*10) and vertical){ line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);cy=x0;} //vertical
		if(( theta>CV_PI/180*80 && theta<CV_PI/180*100) and horizontal){ 
			if(y0>ymin){
				line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
				ymin=y0;
			}
		cx=ymin;
		} //horizontal

	}
	
	return;
}
