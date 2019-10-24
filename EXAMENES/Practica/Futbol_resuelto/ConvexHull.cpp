#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

Mat src; Mat src_gray,original,drawing;
int thresh2 = 100;
int max_thresh2 = 255;
RNG rng2(12345);

/// Function header
void thresh_callback(int, void* );

/** @function main */
void ConvexHull(cv::Mat src,int threshold_ch,cv::Mat &output){
	/// Load source image and convert it to gray
	cv::Mat segment,mask,mask_inv;
	original=src.clone();
	/// Convert image to gray and blur it
	cvtColor( src, src_gray, CV_BGR2GRAY );
	blur( src_gray, src_gray, Size(3,3) );
	Mat src_copy = src.clone();
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	
	/// Detect edges using Threshold
	threshold( src_gray, threshold_output, threshold_ch, 255, THRESH_BINARY );
	
	/// Find contours
	findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	
	/// Find the convex hull object for each contour
	vector<vector<Point> >hull( contours.size() );
	for( int i = 0; i < contours.size(); i++ )
	{  convexHull( Mat(contours[i]), hull[i], false ); }
	
	/// Draw contours + hull results
	drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
	Mat drawing2 = Mat::zeros( threshold_output.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng2.uniform(0, 255), rng2.uniform(0,255), rng2.uniform(0,255) );
		//drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		drawContours( drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	}
	double a=0,ac,idxmax;
	for( int i = 0; i < contours.size(); i++ ){
		ac=contourArea(Mat(contours[i]));
		if(ac>a){
			a=ac;idxmax=i;
		}
	}
	Scalar color = Scalar( rng2.uniform(0, 255), rng2.uniform(0,255), rng2.uniform(0,255) );
	drawContours( drawing2, hull, idxmax, color, 1, 8, vector<Vec4i>(), 0, Point() );
	output=drawing.clone();
}
void ConvexHull(cv::Mat src){
	/// Load source image and convert it to gray
	cv::Mat segment,mask,mask_inv;
	original=src.clone();
	/// Convert image to gray and blur it
	cvtColor( src, src_gray, CV_BGR2GRAY );
	blur( src_gray, src_gray, Size(3,3) );
	
	/// Create Window
	char* source_window = "Source";
	namedWindow( source_window, CV_WINDOW_AUTOSIZE );
	imshow( source_window, src );
	
	createTrackbar( " Threshold:", "Source", &thresh2, max_thresh2, thresh_callback );
	thresh_callback( 0, 0 );
	waitKey(0);
	return;
}

/** @function thresh_callback */
void thresh_callback(int, void* )
{
	Mat src_copy = src.clone();
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	
	/// Detect edges using Threshold
	threshold( src_gray, threshold_output, thresh2, 255, THRESH_BINARY );
	
	/// Find contours
	findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	
	/// Find the convex hull object for each contour
	vector<vector<Point> >hull( contours.size() );
	for( int i = 0; i < contours.size(); i++ )
	{  convexHull( Mat(contours[i]), hull[i], false ); }
	
	/// Draw contours + hull results
	drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
	Mat drawing2 = Mat::zeros( threshold_output.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
	{
		Scalar color = Scalar( rng2.uniform(0, 255), rng2.uniform(0,255), rng2.uniform(0,255) );
		//drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		drawContours( drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
	}
	double a=0,ac,idxmax;
	for( int i = 0; i < contours.size(); i++ ){
		ac=contourArea(Mat(contours[i]));
		if(ac>a){
			a=ac;idxmax=i;
		}
	}
	Scalar color = Scalar( rng2.uniform(0, 255), rng2.uniform(0,255), rng2.uniform(0,255) );
	drawContours( drawing2, hull, idxmax, color, 1, 8, vector<Vec4i>(), 0, Point() );
	/// Show in a window
	namedWindow( "Contours and Hull", CV_WINDOW_AUTOSIZE );
	imshow( "Contours and Hull", drawing );
	imshow( "Max Hull over image", drawing2+original);
	imshow( "Max Hull over segment", drawing2+src);
}
