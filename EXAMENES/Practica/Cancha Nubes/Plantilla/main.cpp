#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"

using namespace cv;
using namespace std;
using namespace pdi;

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


Mat segmentarColor(int x, int y,int v1x1,int v1y1, Mat img){
	Mat imgMasc = img.clone();
	
	//ESFERA
	int aux1 = max(v1x1,x), aux2 = max(v1y1,y);
	v1x1 = min(v1x1,x); v1y1 = min(v1y1,y);
	int dyEsf =  aux2-v1y1, dxEsf =  aux1-v1x1;
	Mat esfera = Mat::zeros(dyEsf,dxEsf,CV_8UC3);
	esfera = imgMasc(Rect(v1x1,v1y1,dxEsf,dyEsf)).clone();
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

Mat sumaMatricesPromedio2(vector<Mat> imagenes){
	int nImagenes = imagenes.size();
//	int nImagenes = 3;
	
	Mat suma = Mat::zeros(imagenes[0].rows,imagenes[0].cols,imagenes[0].type());
	
	int pixel=0;
	int cont=0;

		//			cout<<imagen.rows<<" "<<imagen.cols<<endl;
		for (int i=0;i<imagenes[0].rows;i++){
			
			for (int j=0;j<imagenes[0].cols;j++){
				cont=0;
//				Vec3b pixelSuma = suma.at<Vec3b>(i,j);
				Vec3b pixelSuma =Vec3b(0,0,0);
				int a=0,b=0,c=0;
				for (int z=0;z<nImagenes;z++){
					
//					Mat imagen = imagenes[z].clone();
			
					
//					Vec3b pixelImg = imagen.at<Vec3b>(i,j);
//					imagenes[z].at<Vec3b>(i,j)[0]
					if(imagenes[z].at<Vec3b>(i,j)[0]== 0 && imagenes[z].at<Vec3b>(i,j)[1]== 0 && imagenes[z].at<Vec3b>(i,j)[2]== 0)
						cont++;		
//					
	
					a+=(int)imagenes[z].at<Vec3b>(i,j)[0];
					b+=(int)imagenes[z].at<Vec3b>(i,j)[1];
					c+=(int)imagenes[z].at<Vec3b>(i,j)[2];
				}	
//				cout<<endl;
				if (cont==nImagenes)
					cont=nImagenes-1;
				
				suma.at<Vec3b>(i,j)[0] = a/(nImagenes-cont);//nImagenes;
				suma.at<Vec3b>(i,j)[1] = b/(nImagenes-cont);//nImagenes;
				suma.at<Vec3b>(i,j)[2] = c/(nImagenes-cont);//nImagenes;
				
				
			}
		}
	
	return suma;
}
Mat EcualizaEnRGB(Mat img){
	Mat res;
	vector<Mat> x;
	split(img,x);
	equalizeHist(x[0],x[0]);
	equalizeHist(x[1],x[1]);
	equalizeHist(x[1],x[1]);
	merge(x,res);
	return res;
}
int main(int argc, char** argv) {
	Mat imgParaSegm = imread("Imagenes/10.jpg");
	imshow("Para segmentar",imgParaSegm);
	Mat imgOriginal = imgParaSegm.clone();
	namedWindow("Original",CV_WINDOW_KEEPRATIO);
	imshow("Original",imgOriginal);

	
	
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
		imgs[i] = auxMat.clone();

	}
	int f=imgs[0].rows, c=imgs[0].cols;
	Mat rectangulo=imgs[1](Rect(398,312,75,43));

	
	//segmentado rgb-------------------
	vector <Mat> imgs2(imgs.size());
	
	for(int i=0;i<imgs.size();i++){
	Mat imgSeg = segmentarColor(rectangulo,imgs[i]);
//	Mat ker = (Mat_<uchar>(5,5) <<   
//			   0,0,1,0,0,
//			   0,1,1,1,0,
//			   1,1,1,1,1,
//			   0,1,1,1,0,
//			   0,0,1,0,0);
	int size=3;
	Mat ee = getStructuringElement( MORPH_CROSS,
								   Size( 2*size + 1, 2*size+1 ),
								   Point( -1, -1 ) );
	
	dilate(imgSeg,imgSeg,ee,Point(-1,-1),2);
	Mat segmentado;
	imgs[i].copyTo(segmentado,imgSeg);
	imgs2[i]=imgs[i].clone()-segmentado;
	}
	
	
	//segmentado hsv------------
////	Mat imgHSV;
////	cvtColor(imgs[1],imgHSV,CV_BGR2HSV);
////	imshow("HSV",imgHSV);

////	Mat pruebaSeg=SegmentarPorColor(imgs[1], 400, 310, 81, 48,
////						  0,0,1,0);
////	info(pruebaSeg);
////	info(imgs[1]);
////	Mat segmentado;
////	imgs[1].copyTo(segmentado,pruebaSeg);
////	imshow("skjdfjhfgghdkkj",imgs[1]-segmentado);
Mat resultado = sumaMatricesPromedio2(imgs2);
//	/*info(resultado);*/
//	imshow("Borrar",resultado);
	
	//ROTO LA IMAGEN PROMEDIADA CON EL ANGULO OBTENIDO ANTES MEDIANTE HOUGH
/*	resultado = rotate(resultado,(theta1*180)/CV_PI+180);*/
/**	cout<<(theta1*180)/CV_PI<<endl<<"rho: "<<rho1;*/
//	vector<Mat>imgBGR(3);
//	split(resultado,imgBGR);
//	Mat histo;
//	histo=histograma(imgBGR[0],255);
//	imshow("hi",histo);
//	merge(imgBGR,resultado);
	
	imshow("Original",resultado);
	setMouseCallback("Original",valorPixel,&resultado);
	
//SEGMENTO TRIBUNA
imgParaSegm = segmentarColor(305,335,320,293,resultado);
erode(imgParaSegm,imgParaSegm,Mat(),Point(-1,-1),2);
/*imgParaSegm = segmentarColor(386,284,390,274,resultado);*/
imshow("Segmentada",imgParaSegm);

//HOUGH------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
vector<Vec2f> lines1;
HoughLines(imgParaSegm, lines1, 1, CV_PI/180, 150, 0, 0);
Mat paraHough = imgOriginal.clone();
int s1 = lines1.size();
float rho1 = lines1[2][0], theta1 = lines1[2][1];
Point pt11, pt21;
double a1 = cos(theta1), b1 = sin(theta1);
double x01 = a1*rho1, y01 = b1*rho1;
pt11.x = cvRound(x01 + 1000*(-b1));
pt11.y = cvRound(y01 + 1000*(a1));
pt21.x = cvRound(x01 - 1000*(-b1));
pt21.y = cvRound(y01- 1000*(a1));
line( paraHough, pt11, pt21, Scalar(0,0,255), 1, CV_AA);

imshow("Hough",paraHough);
paraHough = rotate(paraHough,(theta1*180)/CV_PI+180);
imshow("Hough1",paraHough);



	namedWindow("Final",CV_WINDOW_KEEPRATIO);

	imshow("Final",resultado);
	
	waitKey(0);
	return 0;
} 
