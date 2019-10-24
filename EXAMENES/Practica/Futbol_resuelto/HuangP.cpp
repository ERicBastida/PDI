#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;


/// Global variables

/** General variables */
Mat source, edges;
Mat src_gray2;
Mat standard_hough, probabilistic_hough;
int min_threshold = 50;
int max_trackbar = 150;

const char* standard_name = "Standard Hough Lines Demo";
const char* probabilistic_name = "Probabilistic Hough Lines Demo";

int s_trackbar = max_trackbar;
int p_trackbar = max_trackbar;

/// Function Headers
void Standard_Hough( int, void* );
void Probabilistic_Hough( int, void* );

/**
* @function main
*/
void HuangP(cv::Mat im){
	//Inputs:
	// YOU CAN MOVE the Canny threshold also
	// Im: image
	
	
	/// Read the image
	source = im.clone();
	
	/// Pass the image to gray
	cvtColor( source, src_gray2, COLOR_RGB2GRAY );
	
	/// Apply Canny edge detector
	Canny( src_gray2, edges, 50, 200, 3 );
	
	/// Create Trackbars for Thresholds
	char thresh_label[50];
	sprintf( thresh_label, "Thres: %d + input", min_threshold );
	
	namedWindow( standard_name, WINDOW_AUTOSIZE );
	createTrackbar( thresh_label, standard_name, &s_trackbar, max_trackbar, Standard_Hough);
	
	namedWindow( probabilistic_name, WINDOW_AUTOSIZE );
	createTrackbar( thresh_label, probabilistic_name, &p_trackbar, max_trackbar, Probabilistic_Hough);
	
	/// Initialize
	Standard_Hough(0, 0);
	Probabilistic_Hough(0, 0);
	waitKey(0);
	return;
}
/**
* @function Standard_Hough
*/
void Standard_Hough( int, void* )
{
	vector<Vec2f> s_lines;
	cvtColor( edges, standard_hough, COLOR_GRAY2BGR );
	
	/// 1. Use Standard Hough Transform
	HoughLines( edges, s_lines, 1, CV_PI/180, min_threshold + s_trackbar, 0, 0 );
	
	/// Show the result
	for( size_t i = 0; i < s_lines.size(); i++ )
	{
		float r = s_lines[i][0], t = s_lines[i][1];
		double cos_t = cos(t), sin_t = sin(t);
		double x0 = r*cos_t, y0 = r*sin_t;
		double alpha = 1000;
		
		Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
		Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
		line( standard_hough, pt1, pt2, Scalar(255,0,0), 3, CV_AA);
	}
	
	imshow( standard_name, standard_hough );
}

/**
* @function Probabilistic_Hough
*/
void Probabilistic_Hough( int, void* )
{
	vector<Vec4i> p_lines;
	cvtColor( edges, probabilistic_hough, COLOR_GRAY2BGR );
	
	/// 2. Use Probabilistic Hough Transform
	HoughLinesP( edges, p_lines, 1, CV_PI/180, min_threshold + p_trackbar, 30, 10 );
	
	/// Show the result
	for( size_t i = 0; i < p_lines.size(); i++ )
	{
		Vec4i l = p_lines[i];
		line( probabilistic_hough, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, CV_AA);
	}
	
	imshow( probabilistic_name, probabilistic_hough );
}

void HuangP(cv::Mat im,cv::Mat &cdst,cv::Mat &dst,double threshold,double angle,double thresholdAngle){
	//Inputs:
	// YOU CAN MOVE minLineLength and maxLineGap in line of HoughLinesP
	// YOU CAN MOVE the Canny threshold also
	// Im: image
	// threshold: for HuangLines 
	// angle: angle of lines [0 180]
	// thresholdAngle: threshold for detect lines in angles [angle-thresholdAngle<lines<angle+thresholdAngle ]
	// vertical and horizontal: for detect vertical or horizontal lines
	//Outputs:
	// cdst: image and HoughLines
	// dst: contour of image

	Mat src = im.clone();
	Point2f pt1, pt2, v;
	float aux,magnitud_2;
	float rho,theta;
	double C1,C2,T=angle*CV_PI/180,epsilon_theta=thresholdAngle*CV_PI/180;
	double min_T=T-epsilon_theta;
	double max_T=T+epsilon_theta;
	Canny(src, dst, 50, 200, 3); //MOVE THIS PARAMETERS IF YOU NEED
	cvtColor(dst, cdst, CV_GRAY2BGR);
	
	vector<Vec4i> lines;
	//HoughLinesP(InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0 )
	HoughLinesP(dst, lines, 1, CV_PI/180, threshold, 50, 10 );
	for( size_t i = 0; i < lines.size(); i++ )
	{	
		Vec4i l = lines[i];
		pt1=Point2f(lines[i][0],lines[i][1]);
		pt2=Point2f(lines[i][2],lines[i][3]);
		v=pt2-pt1;
		/** Rotacion a 90 grados en sentido horario (renegue hasta darme cuenta con esto) **/
		aux=v.y;
		v.y=-v.x;
		v.x=aux;
		magnitud_2=v.x*v.x+v.y*v.y;
		theta=acos(v.x/(sqrt(magnitud_2)));
		C1=(abs(theta-CV_PI/2.0));
		C2=(abs(theta-CV_PI/2.0));
		if(C1<=max_T && C2>= min_T){
		line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
		}
	}

	return;
}
