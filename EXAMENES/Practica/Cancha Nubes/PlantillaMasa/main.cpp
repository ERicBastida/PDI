#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"

int v1x=-1,v1y=0;
bool bandera=false;

//SUMA LAS IMAGENES DE UN VECTOR Y COLOCA EL RESULTADO AL FINAL
vector<Mat> suma(vector<Mat> v){
	vector<Mat> retorno(v.size()+1);
	Mat resultado = Mat::zeros(v[0].rows,v[0].cols,CV_8UC1);
	for (int k=0;k<v.size();k++){
		Mat imagen = v[k];
		retorno[k]=v[k];
		for (int i=0;i<imagen.rows;i++){
			for (int j=0;j<imagen.cols;j++){
				resultado.at<uchar>(i,j) += imagen.at<uchar>(i,j)/v.size();
			}
		}
	}
	retorno[retorno.size()-1]=resultado;
	return retorno;
}

//MouseCallback -> DEVUELVE EL VALOR DEL PIXEL
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

void segmentarColor1(int event, int x, int y, int, void* userData){
	
	if( event != EVENT_LBUTTONDOWN )
		return;
	
	if (v1x==-1){
		v1x=x;v1y=y;
	}else{
		Mat img = *( static_cast<Mat*>(userData) ); //puede ser lo que sea, adentro le digo q es un matriz
		Mat imgMasc = img.clone();
		
		//ESFERA
		int aux1 = max(v1x,x), aux2 = max(v1y,y);
		v1x = min(v1x,x); v1y = min(v1y,y);
		int dyEsf =  aux2-v1y, dxEsf =  aux1-v1x;
		Mat esfera = Mat::zeros(dyEsf,dxEsf,CV_8UC3);
		esfera = imgMasc(Rect(v1x,v1y,dxEsf,dyEsf)).clone();
		
		cout<<"COORDENADAS: "<<endl;
		cout<<"Punto 1: "<<v1x<<" "<<v1y<<endl;
		cout<<"Punto 2: "<<aux1<<" "<<aux2<<endl;
		
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
		
		Mat segmentarColor = Mat::zeros(imgMasc.rows,imgMasc.cols,CV_8UC1);
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
					segmentarColor.at<uchar>(i,j) = 0;
				else
					segmentarColor.at<uchar>(i,j) = 255;
			}
		}
		
		namedWindow("Imagen Fragmentada en RGB",CV_WINDOW_KEEPRATIO);
		imshow("Imagen Fragmentada en RGB",segmentarColor);		
		v1x=-1;
		
		bandera=true;
	}
	
	return;
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
	Mat imgParaSegm = imread("Imagenes/10.jpg");
	imshow("Para segmentar",imgParaSegm);
	Mat imgOriginal = imgParaSegm.clone();
	namedWindow("Original",CV_WINDOW_KEEPRATIO);
	imshow("Original",imgOriginal);
	setMouseCallback("Original",valorPixel,&imgOriginal);
	
	//SEGMENTO TRIBUNA
	/*imgParaSegm = segmentarColor(305,335,320,293,imgParaSegm);*/
	imgParaSegm = segmentarColor(386,284,390,274,imgParaSegm);
	imshow("Segmentada",imgParaSegm);
	
	//HOUGH------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	vector<Vec2f> lines1;
	HoughLines(imgParaSegm, lines1, 1, CV_PI/180, 80, 0, 0);
	Mat paraHough = imgOriginal.clone();
	int s1 = lines1.size();
	float rho1 = lines1[0][0], theta1 = lines1[0][1];
	Point pt11, pt21;
	double a1 = cos(theta1), b1 = sin(theta1);
	double x01 = a1*rho1, y01 = b1*rho1;
	pt11.x = cvRound(x01 + 1000*(-b1));
	pt11.y = cvRound(y01 + 1000*(a1));
	pt21.x = cvRound(x01 - 1000*(-b1));
	pt21.y = cvRound(y01- 1000*(a1));
	line( paraHough, pt11, pt21, Scalar(0,0,255), 1, CV_AA);
	
	paraHough = rotate(paraHough,(theta1*180)/CV_PI+180);
	imshow("Hough1",paraHough);
	
	//PROMEDIO IMAGENES
	String cadena = "Imagenes/X.jpg";
	int l = cadena.size();
	char aux[l];
	int cantImg = 15;
	vector<Mat> imgs(cantImg);
	for (int i=0;i<cantImg;i++){
		sprintf(aux,"Imagenes/%d.jpg",i+1);
		cout<<aux<<endl;
		Mat auxMat = imread(aux);
		imgs[i] = auxMat;
	}
	int f=imgs[0].rows, c=imgs[0].cols;
	Mat resultado = sumaMatricesPromedio(imgs);
	imshow("Borrar",resultado);
	
	//ROTO LA IMAGEN PROMEDIADA CON EL ANGULO OBTENIDO ANTES MEDIANTE HOUGH
	resultado = rotate(resultado,(theta1*180)/CV_PI+90);
	namedWindow("Final",CV_WINDOW_KEEPRATIO);
	imshow("Final",resultado);
	
	waitKey(0);
	return 0;
} 
