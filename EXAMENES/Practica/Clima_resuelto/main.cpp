#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"
#include <opencv2/core/core.hpp>


using namespace cv;
using namespace std;
using namespace pdi;


Point2d EncontrarBdZ(Mat img){
	Point2d p1;
	for(int i=0; i<img.rows;i++){
		for(int j=0; j<img.cols; j++){
			int v = (int)img.at<uchar>(i,j);
			if (v != 0){
				p1.x = j;
				p1.y = i;
				return p1;
			}
		}
	}
}
float BdzBarra(Mat img,int B,int G,int R){
	int col = img.cols/2;
	vector<Mat> canales;
	split(img,canales);
	int tol = 2;
	float Bdz = -1.2;
	for (int i=0;i<img.rows;i++){
		int Bb = (int)canales[0].at<uchar>(i,col);
		int Gg = (int)canales[1].at<uchar>(i,col);
		int Rr = (int)canales[2].at<uchar>(i,col);
		cout << i << " " << Bb << " " << Gg << " " << Rr << endl;
		if (tol+Bb>=B && B>=Bb-tol){
			if (tol+Gg>=G && G>=Gg-tol){
				if (tol+Rr>=R && R>=Rr-tol){
					Bdz = (68*340-68*i)/340;
					cout <<"ADetri" << i<< " " << img.rows;
					return Bdz;
				}
			}
		}
		
	}
}


int main(int argc, char** argv) {
	Mat img = imread("4.jpg",IMREAD_COLOR);
	info(img);
	
	
//	cvtColor(img,img,COLOR_BGR2HSV);

//	vector<Mat> canales(3);
//	split(img,canales);
//	imshow("H",canales[0]);
//	imshow("S",canales[1]);
	Mat img2 = img(Rect(85,30,345,345));
	img2 = SegmentacionHSV(img2,150,255,150,255);
//	cvtColor(img2,img2,COLOR_BGR2GRAY);
//	threshold(img2,img2,0,255,THRESH_BINARY);
	
	Mat ee = getStructuringElement(MORPH_RECT,Size(2,2));
	morphologyEx(img2,img2,MORPH_ERODE,ee);
	ee = getStructuringElement(MORPH_RECT,Size(3,3));
	morphologyEx(img2,img2,MORPH_DILATE,ee);
	
	cvtColor(img2,img2,COLOR_BGR2HSV);
	vector<Mat> canales(3);
	split(img2,canales);
	
	Point2d c = EncontrarBdZ(canales[0]);
	circle(img2,c,15,150,2);
	
	Mat img3 = img(Rect(85,30,345,345));
	cvtColor(img3,img3,COLOR_HSV2BGR);
	split(img3,canales);
	int B = (int)canales[0].at<uchar>(c.x,c.y);
	int G = (int)canales[1].at<uchar>(c.x,c.y);
	int R = (int)canales[2].at<uchar>(c.x,c.y);
	cout << "assss" << B << " " << G << " " << R << endl;
	
	Mat barra = img(Rect(460,35,7,340));
	float Bdz = BdzBarra(barra,B,G,R);
	cout << Bdz;
	
//	namedWindow("Imagen1");
//	imshow("Imagen1",img);
	namedWindow("Imagen");
	imshow("Imagen",img);
	waitKey(0);
	return 0;
} 
