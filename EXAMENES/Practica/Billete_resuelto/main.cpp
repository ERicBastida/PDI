#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"

using namespace cv;
using namespace pdi;
using namespace std;

int EstaDerecha(Mat img){
	int q = 0;
	for(int i=0; i<img.rows; i++){
		for(int j=0; j<img.cols; j++){
			int p = (int)img.at<uchar>(i,j);
			if (p == 0){
				q = 1;
			}				
		}
	}
	return q;
}

int ContarDiam(Mat img){
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	return contours.size();
}



int main(int argc, char** argv) {
	Mat img = imread("B100.jpg");
	info(img);
	
	/** Vamos a descubrir si esta rotada o no **/
	Mat imgBin;
	cvtColor(img,imgBin,COLOR_BGR2GRAY);
	threshold(imgBin,imgBin,150,255,THRESH_BINARY);
	Mat rect = imgBin(Rect(125,150,50,50));
	threshold(rect,rect,0,255,THRESH_BINARY);
	
	int r = EstaDerecha(rect);
//	cout << r;
	
	if (r == 1){ /// /// La imagen no esta derecha, hay rotarla 180 grados
		/** Nos posicionamos en la parte de los rombos **/
		Mat imgBinR = rotate(imgBin,180);
		
		Mat imgfinalR = imgBinR(Rect(130,10,60,100));
//		threshold(imgfinalR,imgfinalR,0,255,THRESH_BINARY);
		
		Mat ee = getStructuringElement(MORPH_RECT,Size(5,5));
		morphologyEx(imgfinalR,imgfinalR,MORPH_CLOSE,ee);
		
		int d = ContarDiam(imgfinalR);
//		cout << "diamentes " << d;
		if (d-1 == 6) { cout << " El billete es de 2 pesos ";}
		if (d-1 == 5) { cout << " El billete es de 5 pesos ";}
		if (d-1 == 4) { cout << " El billete es de 10 pesos ";}
		if (d-1 == 3) { cout << " El billete es de 20 pesos ";}
		if (d-1 == 2) { cout << " El billete es de 50 pesos ";}
		if (d-1 == 1) { cout << " El billete es de 100 pesos ";}
		
		
	}else {
		Mat imgfinal = imgBin(Rect(125,0,55,100));
		threshold(imgfinal,imgfinal,0,255,THRESH_BINARY);
		
		Mat ee = getStructuringElement(MORPH_RECT,Size(5,5));
		morphologyEx(imgfinal,imgfinal,MORPH_CLOSE,ee);
		
		int d = ContarDiam(imgfinal);
		//		cout << "diamentes " << d;
		if (d-1 == 6) { cout << " \nEl billete es de 2 pesos ";}
		if (d-1 == 5) { cout << " \nEl billete es de 5 pesos ";}
		if (d-1 == 4) { cout << " \nEl billete es de 10 pesos ";}
		if (d-1 == 3) { cout << " \nEl billete es de 20 pesos ";}
		if (d-1 == 2) { cout << " \nEl billete es de 50 pesos ";}
		if (d-1 == 1) { cout << " \nEl billete es de 100 pesos ";}
		
		
	}
	
	
	namedWindow("Imagen",0);
	imshow("Imagen",img);
	
	
	waitKey(0);
	return 0;
} 
