#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

void RoI(cv::Mat im,cv::Mat &roi,int x, int y, int w, int h){
	//Inputs:
	// im: image
	// x: coordinate x superior
	// y: coordinate y superior
	// w: width
	// h: height
	
	//Output:
	// roi: image cutted
	
	if ( im.empty() ) { std::cerr<<"imagen vacia";}
	roi = im( Rect(x,y,w,h) );
	return;
}
