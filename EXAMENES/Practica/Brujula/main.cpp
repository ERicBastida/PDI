#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"
using namespace cv;
using namespace std;
using namespace pdi;


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




Mat segmentarColor(Mat rectangulo, Mat img){
	Mat imgMasc = img.clone();
	
	//ESFERA
	
	int dyEsf = rectangulo.rows;
	int dxEsf =  rectangulo.cols;
	Mat esfera = rectangulo.clone();
	imshow("esfera",esfera);
	
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

Mat SegmentarPorColor(Mat img,Mat aux,
					  bool op,int tol0=0, int tol1=0, int tol2=0){
	
	imshow("aux",aux);
	Mat HSVimg;cvtColor(img,HSVimg,CV_BGR2HSV);
	Mat HSVaux;cvtColor(aux,HSVaux,CV_BGR2HSV);
	
	int m0i,m0f,m1i,m1f,m2i,m2f;
	SacarDatos(HSVaux,m0i,m0f,m1i,m1f,m2i,m2f);
	
	Mat segmentado=Mat::zeros(HSVimg.size(),HSVimg.type());
	inRange(HSVimg,Scalar(m0i-tol0,m1i-tol1,0),Scalar(m0f+tol0,m1f+tol1,255),HSVaux);
	
	
	HSVimg.copyTo(segmentado,HSVaux);
	cvtColor(segmentado,segmentado,CV_HSV2BGR);
	//		imshow("Segmentado HSV",segmentado);
	return HSVaux;
	
	
}

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


int main(int argc, char** argv) {
	
	Mat img=imread("2.png");
//	bool carga=false;
//	//Mat seg=SegmentarPorColor(img,160,210,9,3,0,1,0,0);
//	//	Mat seg=SegmentarPorColor(img,160,210,9,3,0,10,1,0);
//	if(carga){
//		
//		
//	}else{
//		
//		Mat cuad1=img(Rect(160,210,10,5));
//		Mat cuad2=img(Rect(201,33,2,2));
//		
//		imwrite("cuad1.jpg",cuad1);
//		imwrite("cuad2.jpg",cuad2);
//	}
	Mat cua1=imread("cuad1.jpg");
	Mat cua2=imread("cuad2.jpg");
//	Mat masc_roja=segmentarColor(160,210,170,215,img);
//	Mat masc_norte=segmentarColor(201,33,203,35,img);
//	
	Mat masc_roja=segmentarColor(cua1,img);
	Mat masc_norte=segmentarColor(cua2,img);

	
	
	
	
	
//	imshow("imagen original",img);
	imshow("Flecha Segmentada",masc_roja);
	imshow("Norte Segmentado",masc_norte);
	masc_roja=masc_norte;
	
	Mat ker = (Mat_<uchar>(5,5) <<   0,0,1,0,0,
			   0,1,1,1,0,
			   1,1,1,1,1,
			   0,1,1,1,0,
			   0,0,1,0,0);
	
	
	//	dilate(masc_roja,masc_roja,ker,Point(-1,-1),8);
	dilate(masc_roja,masc_roja,Mat(),Point(-1,-1),11);
	
	erode(masc_roja,masc_roja,Mat(),Point(-1,-1),9);
	
	imshow("Dilatada",masc_roja);
	
	
	//	//	//HOUGH---------------------------------------------
	vector<Vec2f> lines;
	
	HoughLines(masc_roja, lines, 1, CV_PI/180, 40, 0, 0);
	int s = lines.size();
	cout<<s<<endl;
	
	float rho = lines[1][0], theta = lines[1][1];
	Point pt1, pt2,centro;
	double a = cos(theta), b = sin(theta);
	double x0 = a*rho, y0 = b*rho;
	pt1.x = cvRound(x0 + 1000*(-b));
	pt1.y = cvRound(y0 + 1000*(a));
	pt2.x = cvRound(x0 - 1000*(-b));
	pt2.y = cvRound(y0 - 1000*(a));
	centro.x=img.rows/2;
	centro.y=img.cols/2;
	bool st=false;
	for(int i=0;i<img.cols;i++){
//		if(masc_roja.at<uchar>(masc_roja.rows/2,i)==255){
		if(masc_roja.at<uchar>(215,i)==255){
			cout<<"para abajo"<<endl;
			pt2=pt1;
			st=true;
			break;
		}
	}
		
//	if(st)
	
	
	
	
	//	line( img, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
	line( img, centro, pt2, Scalar(0,0,255), 3, CV_AA);
//	line( img, centro, pt1, Scalar(0,0,255), 3, CV_AA);
	
	
	
	/**ACA TRABAJO EL NORTE**/
	Mat norte=masc_norte.clone();
	erode(norte,norte,ker,Point(-1,-1),3);
	erode(norte,norte,Mat(),Point(-1,-1),1);
	
	imshow("Buscando el norte",norte);
	int ib=0,jb=0;
	
	for(int i=0;i<norte.rows;i++){
		for(int j=0;j<norte.cols;j++){
			if(norte.at<uchar>(i,j)==255){
				ib=i;
				jb=j;
				break;
			}
		}}
	cout<<"Encontro "<<ib<<" "<<jb<<endl;
	Point N;
	N.x=jb;
	N.y=ib;
	line( img, centro, N, Scalar(0,255,0), 3, CV_AA);
	
	
	Point v1;
	v1.x=N.x-centro.x;
	v1.y=centro.y-N.y;
	
//	if(st){pt1=pt2;};
	Point v2;
	v2.x=pt1.x-centro.x;
	v2.y=centro.y-pt1.y;
	
	
	cout<<"Vectores"<<endl;
	cout<<v1<<endl;
	cout<<v2<<endl;
	
	
	float normaV1=sqrt(v1.x*v1.x +v1.y*v1.y);
	float normaV2=sqrt(v2.x*v2.x +v2.y*v2.y);
	Point u1;
	Point u2;
	float v1x=v1.x/normaV1;
	float v1y=v1.y/normaV1;
	
	float v2x=v2.x/normaV2;
	float v2y=v2.y/normaV2;
	
	
	cout<<"["<<v1x<<"," <<v1y<<"]"<<endl;
	cout<<"["<<v2x<<"," <<v2y<<"]"<<endl;
	
	
	cout<<"norma v1"<<normaV1<<endl;
	cout<<"norma v2"<<normaV2<<endl;
	float aux1=(v1x*v2x+v1y*v2y);
	cout<<"cos-1 de "<<aux1<<endl;
	
	float aa[2] = {(float)v1.x,(float) v1.y};
	float bb[2] = {(float)v2.x, (float)v2.y};
	
	
	cv::Mat AA(1,2,CV_32FC1,aa);
	cv::Mat BB(1,2,CV_32FC1,bb);
	
	
	cout << AA << endl;
	cout << BB << endl;
	cout << AA.dot(BB) << " should be equal to 11" << endl;
	if(AA.dot(BB)>0){
		float ang=acos(aux1);
		cout<<"angulo de: "<<int((ang*180)/3.141516)<<"°"<<endl;
	}else{
		float ang=acos(aux1);
		cout<<"angulo "<<int(360.0-(ang*180)/3.141516)<<"°"<<endl;
	}
	
	
	
	
	
	//	
	//	 rho = lines[4][0], theta = lines[4][1];
	//	 a = cos(theta), b = sin(theta);
	//	 x0 = a*rho; y0 = b*rho;
	//	pt1.x = cvRound(x0 + 1000*(-b));
	//	pt1.y = cvRound(y0 + 1000*(a));
	//	pt2.x = cvRound(x0 - 1000*(-b));
	//	pt2.y = cvRound(y0 - 1000*(a));
	//	centro.x=img.rows/2;
	//	centro.y=img.cols/2;
	////	line( img, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
	//	line( img, centro, pt1, Scalar(0,255,0), 3, CV_AA);
	imshow("imagen original Linea",img);
	
	waitKey(0);
	return 0;
} 
