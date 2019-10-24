#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"

using namespace cv;
using namespace pdi;
using namespace std;

Mat etiq,imgReg;
int etiqueta;
int tam;

void crecer(int i,int j){
	tam++;
	etiq.at<uchar>(i,j) = etiqueta;
	for(int k = -1;k<2;k++){
		for(int m = -1;m<2;m++){
			int ni=i+k;
			int nj=j+m;
			if( imgReg.at<uchar>(ni,nj)!=0 and etiq.at<uchar>(ni,nj)==0 and ni!=nj){
				crecer(ni,nj);
			}
		}
	}
}

vector<Point2d> etiquetar_regiones_no_solapadas(){
	etiqueta=1;
	int monedas =0;
	vector<Point2d> v;
	for(int i = 0 ; i<imgReg.rows;i++){
		for(int j = 0 ; j<imgReg.cols*3;j++){
			if( (int)imgReg.at<uchar>(i,j)!= 0 && (int)etiq.at<uchar>(i,j)==0 ){
				tam =0;
				crecer(i,j);
				if (tam<200){
					cout<<"Tamaño: "<<tam<<endl;
					v.push_back(Point2d((int)j/3,i));
				}
				etiqueta++;
			}
		}
	}
	//cout<<"Cantidad de monedas: "<<monedas<<endl;
	return v;
}


int ContarManzanas(Mat img){
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	return contours.size();
}
int CalcularTam(Mat roi){
	int c =0;
	for(int i=0; i<roi.rows; i++){
		for(int j=0; j<roi.cols; j++){
			int v = (int)roi.at<uchar>(i,j);
			if (v !=0){
				c += 1;
			}
		}
	}
	return c;
}


int main(int argc, char** argv) {
	Mat img = imread("EXAMEN09.jpg");
	info(img);
	/** CUENTO LA CANTIDAD DE MANZANAS ROJAS  **/
	Mat imgBinM;
	imgBinM = SegmentacionHSV(img, 50, 200, 200, 255);
	cvtColor(imgBinM,imgBinM,COLOR_BGR2GRAY);
	threshold(imgBinM,imgBinM,0,255,THRESH_BINARY);
	
	Mat eeM = getStructuringElement(MORPH_RECT,Size(4,4));
	morphologyEx(imgBinM,imgBinM,MORPH_ERODE,eeM);
	morphologyEx(imgBinM,imgBinM,MORPH_ERODE,eeM);

	
	Mat imgBinMB = imgBinM.clone();
	int mM = ContarManzanas(imgBinMB);
	cout << "\n Manzanas Rojas: " << mM ;
	
	
	/** CUENTO LAS MANZANAS VERDES **/
	Mat imgBinV;
	imgBinV = SegmentacionHSV(img, 30, 50, 200, 245);
	cvtColor(imgBinV,imgBinV,COLOR_BGR2GRAY);
	threshold(imgBinV,imgBinV,0,255,THRESH_BINARY);
	
	Mat eeV = getStructuringElement(MORPH_RECT,Size(4,4));
	morphologyEx(imgBinV,imgBinV,MORPH_ERODE,eeV);
	morphologyEx(imgBinV,imgBinV,MORPH_ERODE,eeV);
	
	Mat imgBinVB = imgBinV.clone();
	int mV = ContarManzanas(imgBinVB);
	cout << "\n Manzanas Verdes: " << mV << endl ;
	
	
	
	imgReg = imgBinM + imgBinV;
		
	cvtColor(imgReg,imgReg, COLOR_GRAY2BGR);
	etiq = Mat::zeros(imgReg.size(),imgReg.type());
	vector<Point2d> v = etiquetar_regiones_no_solapadas();
	
	for(int i=0; i<v.size();i++){
		circle(img,v[i],10,Scalar(255,0,0),2);
	}
	
	namedWindow("Imagen");
	imshow("Imagen",img);
//	namedWindow("Imagen1");
//	imshow("Imagen1",img);
	
	
	
	
	waitKey(0);
	return 0;
} 
