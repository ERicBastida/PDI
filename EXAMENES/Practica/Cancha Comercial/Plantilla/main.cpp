#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"

using namespace cv;
using namespace std;
using namespace pdi;

void valorPixel(int event, int x, int y, int, void* userData){
	if( event != EVENT_LBUTTONDOWN )
		return;
	Mat img = *( static_cast<Mat*>(userData)); //puede ser lo que sea, adentro le digo q es un matriz
	//	info(img);
	Vec3b c = img.at<Vec3b>(y,x);
	int h = c[0], s = c[1], v = c[2];
	cout<<"Punto: "<<x<<" "<<y<<endl;
	cout<<h<<" "<<s<<" "<<v<<"\n";
}


int Media(Mat img){
	int s=0;
	for(int i=0;i<img.rows;i++) { 
		for(int j=0;j<img.cols;j++) { 
			s+=img.at<uchar>(i,j);
		}
	}
	return s/(img.rows*img.cols);
}

void SacarDatos(Mat img,int &m0i, int &m0f,int &m1i, int &m1f,int &m2i, int &m2f   ){
	vector<Mat>x;
	split(img,x);
//	int m0=Media(x[0]);
//	int m1=Media(x[1]);
//	int m2=Media(x[2]);}
	Scalar medias=mean(img);
	int m0=medias[0];
	int m1=medias[1];
	int m2=medias[2];
	
	int std0=0,std1=0,std2=0;
	for(int i=0;i<img.rows;i++) { 
		for(int j=0;j<img.cols;j++) { 
			std0+=pow((m0-x[0].at<uchar>(i,j)),2);
			std1+=pow((m1-x[1].at<uchar>(i,j)),2);
			std2+=pow((m2-x[2].at<uchar>(i,j)),2);
			
		}
	}
	int nn=img.rows*img.cols;
	std0=sqrt(std0/nn);
	std1=sqrt(std1/nn);
	std2=sqrt(std2/nn);
	
	m0i=m0-std0;m1i=m1-std1;m2i=m2-std2;
	m0f=std0+m0;m1f=std1+m1;m2f=std2+m2;
	//	cout<<"Esta entre "<<m0-std0<<" y  "<<std0+m0<<endl;
	//	cout<<"Esta entre "<<m1-std1<<" y  "<<std1+m1<<endl;
	//	cout<<"Esta entre "<<m2-std2<<" y  "<<std2+m2<<endl;
	
	
}

Mat SegmentarPorColor(Mat img, int x, int y, int ancho, int alto,
					  bool op,int tol0=0, int tol1=0, int tol2=0){
	Mat aux=Mat(img,Rect(x,y,ancho,alto));
	imshow("aux",aux);
	if (op){//en RGB
		//	Mat RGBimg;cvtColor(img,RGBimg,CV_HSV2RGB);
		//	Mat RGBaux;cvtColor(aux,RGBaux,CV_HSV2RGB);
		Mat RGBimg=img.clone();
		Mat RGBaux=aux.clone();
		
		int m0i,m0f,m1i,m1f,m2i,m2f;
		SacarDatos(RGBaux,m0i,m0f,m1i,m1f,m2i,m2f);
		
		Mat segmentado=Mat::zeros(RGBimg.size(),RGBimg.type());
		inRange(RGBimg,Scalar(m0i-tol0,m1i-tol1,m2i-tol2),Scalar(m0f+tol0,m1f+tol1,m2f+tol2),RGBaux);
		
		
		//esto es un enmascarado BASTARDO
		//		Rect rectangulo=Rect(5,165,310,260);
		//		Mat msk=Mat::zeros(img.rows,img.cols,CV_8UC1);
		//		rectangle(msk,rectangulo,Scalar(255),CV_FILLED);
		//		vector<Mat>xx;split(RGBaux,xx);
		//		for(int i=0;i<xx[0].rows;i++) { 
		//			for(int j=0;j<xx[0].cols;j++) { 
		//				if ((i<165) || (i>165+260) || (j<5) || (j>315))
		//				{xx[0].at<bool>(i,j)=0;}
		//				
		//			}
		//		}
		//		merge(xx,RGBaux);
		//Hasta aca..
		
		//imshow("asdsa",RGBaux);
		RGBimg.copyTo(segmentado,RGBaux);
		
		//		imshow("Segmentada RGB",segmentado);
		return RGBaux;
		
	}
	else {
		
		Mat HSVimg;cvtColor(img,HSVimg,CV_BGR2HSV);
		Mat HSVaux;cvtColor(aux,HSVaux,CV_BGR2HSV);
		
		int m0i,m0f,m1i,m1f,m2i,m2f;
		SacarDatos(HSVaux,m0i,m0f,m1i,m1f,m2i,m2f);
		
		Mat segmentado=Mat::zeros(HSVimg.size(),HSVimg.type());
		inRange(HSVimg,Scalar(m0i-tol0,m1i-tol1,0),Scalar(m0f+tol0,m1f+tol1,255),HSVaux);
		
	
		
		HSVimg.copyTo(segmentado,HSVaux);
		
		cvtColor(segmentado,segmentado,CV_HSV2BGR);
				imshow("Segmentado HSV",segmentado);
		
		return HSVaux;
	}
	
}

Mat SegmentarPorColor(Mat img,Mat aux,
					  bool op,int tol0=0, int tol1=0, int tol2=0){
	
	imshow("aux",aux);
	if (op){//en RGB
		//	Mat RGBimg;cvtColor(img,RGBimg,CV_HSV2RGB);
		//	Mat RGBaux;cvtColor(aux,RGBaux,CV_HSV2RGB);
		Mat RGBimg=img.clone();
		Mat RGBaux=aux.clone();
		
		int m0i,m0f,m1i,m1f,m2i,m2f;
		SacarDatos(RGBaux,m0i,m0f,m1i,m1f,m2i,m2f);
		
		Mat segmentado=Mat::zeros(RGBimg.size(),RGBimg.type());
		inRange(RGBimg,Scalar(m0i-tol0,m1i-tol1,m2i-tol2),Scalar(m0f+tol0,m1f+tol1,m2f+tol2),RGBaux);
		
		//esto es un enmascarado BASTARDO
		//		Rect rectangulo=Rect(5,165,310,260);
		//		Mat msk=Mat::zeros(img.rows,img.cols,CV_8UC1);
		//		rectangle(msk,rectangulo,Scalar(255),CV_FILLED);
		//		vector<Mat>xx;split(RGBaux,xx);
		//		for(int i=0;i<xx[0].rows;i++) { 
		//			for(int j=0;j<xx[0].cols;j++) { 
		//				if ((i<165) || (i>165+260) || (j<5) || (j>315))
		//				{xx[0].at<bool>(i,j)=0;}
		//				
		//			}
		//		}
		//		merge(xx,RGBaux);
		//Hasta aca..
		
		//imshow("asdsa",RGBaux);
		RGBimg.copyTo(segmentado,RGBaux);
		
		//		imshow("Segmentada RGB",segmentado);
		return RGBaux;
		
	}
	else {
		
		Mat HSVimg;cvtColor(img,HSVimg,CV_RGB2HSV);
		Mat HSVaux;cvtColor(aux,HSVaux,CV_RGB2HSV);
		
		int m0i,m0f,m1i,m1f,m2i,m2f;
		SacarDatos(HSVaux,m0i,m0f,m1i,m1f,m2i,m2f);
		
		Mat segmentado=Mat::zeros(HSVimg.size(),HSVimg.type());
		inRange(HSVimg,Scalar(m0i-tol0,m1i-tol0,0),Scalar(m0f+tol0,m1f+tol1,255),HSVaux);
		
		
		HSVimg.copyTo(segmentado,HSVaux);
		cvtColor(segmentado,segmentado,CV_HSV2RGB);
		//		imshow("Segmentado HSV",segmentado);
		return HSVaux;
	}
	
}


Mat segmentarColor(int x, int y,int v1x1,int v1y1, Mat img){
	Mat imgMasc = img.clone();
	
	//ESFERA
	int aux1 = max(v1x1,x), aux2 = max(v1y1,y);
	v1x1 = min(v1x1,x); v1y1 = min(v1y1,y);
	int dyEsf =  aux2-v1y1, dxEsf =  aux1-v1x1;
	Mat esfera = Mat::zeros(dyEsf,dxEsf,CV_8UC3);
	esfera = imgMasc(Rect(v1x1,v1y1,dxEsf,dyEsf)).clone();
	
	//CENTRO
	vector<float> centro(3), maximos(3);
	centro[0] = 0; centro[1] = 0; centro[2] = 0;
	maximos[0] = 0; maximos[1] = 0; maximos[2] = 0;
	int n = dxEsf*dyEsf;
	for (int i=0;i<dyEsf;i++){
		for (int j=0;j<dxEsf;j++){
			Vec3b pixel = esfera.at<Vec3b>(i,j);
			float v1=(float) pixel[0],v2=(float)  pixel[1],v3=(float) pixel[2];
			centro[0]+=v1; centro[1]+=v2; centro[2]+=v3;
			
			if (v3>maximos[2]) maximos[2]=v3;
			if (v2>maximos[1]) maximos[1]=v2;
			if (v1>maximos[0]) maximos[0]=v1;
			
		}
	}
	centro[0]/=(float)n;centro[1]/=(float)n;centro[2]/=(float)n;
	cout<<"H: "<<centro[0]<<" S: "<<centro[1]<<" I: "<<centro[2]<<"\n";
	
	Mat segmentada = Mat::zeros(imgMasc.rows,imgMasc.cols,CV_8UC1);
	//RADIO DE LA ESFEREA: DISTANCIA DEL CENTRO AL MAXIMO DE CADA CANAL
	float radio = sqrt((maximos[0]-centro[0])*(maximos[0]-centro[0])+
		(maximos[1]-centro[1])*(maximos[1]-centro[1])+
		(maximos[2]-centro[2])*(maximos[2]-centro[2]));
	
	for (int i=0;i<img.rows;i++){
		for (int j=0;j<img.cols;j++){
			Vec3b pixel = imgMasc.at<Vec3b>(i,j);
			float v1 = (float) pixel[0],v2 = (float) pixel[1],v3 = (float) pixel[2];
			float radioAct = sqrt((centro[2]-v3)*(centro[2]-v3)+
				(centro[1]-v2)*(centro[1]-v2)+
				(centro[0]-v1)*(centro[0]-v1));
			if (radioAct>radio)
				segmentada.at<uchar>(i,j) = 0;
			else
				segmentada.at<uchar>(i,j) = 255;
		}
	}
	
	namedWindow("Imagen Fragmentada en RGB",CV_WINDOW_KEEPRATIO);
	imshow("Imagen Fragmentada en RGB",segmentada);		
	
	return segmentada;
}

int main(int argc, char** argv) {
	Mat img=imread("1.png");
	imshow("Original",img);
	Mat segmentada=SegmentarPorColor(img,115, 349, 150, 150,
						  0,10,255,0);
	setMouseCallback("Original",valorPixel,&img);
	imshow("seg",segmentada);
/*	Mat posta=segmentarColor(115, 349,150,150,img);*/
	
	
	waitKey(0);
	return 0;
} 
