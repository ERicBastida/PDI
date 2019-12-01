 #include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"
#include <iostream>
#include <opencv2/core/types_c.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <ostream>
#include "opencv2/core/core.hpp"
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string.h>




using namespace std;
using namespace cv;
using namespace pdi;



//Filtro Media-Alfa Recortado
void filtro_mediaAlfa_recortado(Mat &src, int D,int k_size){
	//k_size: TamaÒo del kernel --> Debe ser IMPAR O EXPLOTA TODO !!
	//Filtro medAlfa Recortado: f(x,y)=[1/(m*n-D)]*sum;
	//D : Cantidad de Elementos que descarto de cada kernel (que sea par !!)
	
	
	int
		Nrows=src.rows,
		Ncols=src.cols,
		R=(k_size-1)/2,
		Mid=(k_size*k_size-1)/2;
	if(D>=Mid){cout<<"Elija D mas pequeno"; return;}
	Mat tmp=src.clone();
	vector <int> V;
	double aux;
	
	for(int i=2; i<Nrows-2; i++){
		for(int j=2; j<Ncols-2 ; j++){
			
			for(int k=i-R; k<=i+R ; k++){  //Recorre el kernel para cada posicion x,y de la imagen
				for(int l=j-R; l<=j+R ; l++){
					V.push_back((int) tmp.at<uchar>(k,l));
				}
			}
			
			sort(V.begin(),V.end());
			aux=0;
			for (int m=Mid-D/2; m<=Mid+D/2; m++){
				aux+=V[m];	}
			aux=aux/(k_size*k_size-D);
			
			
			src.at<uchar>(i,j)= (int) V[Mid];
			V.clear();
		}
		
	}
	
	
}






Mat DibujarHough(Mat img, double nLineas){
	//Canny para detectar los bordes de la imagen
	//cvtColor para cambiar la imagen de escala de grises a BGR y poder mostrar los resultados.
	// nLineas = cantidad minima de lineas a detectar
	Mat dst, cdst;
	Canny(img, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	
	
	double	tita = 1,
		ro = CV_PI/4; // = 1
	//			nlineas = 50; //cantidad minima de lineas a detectar
	
	vector<Vec2f> lines;
	HoughLines(dst, lines,tita,ro, nLineas, 0, 0 );
	
	for ( size_t i = 0; i < lines.size(); i++ ) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
	}
	
	return cdst;
}




Mat DetectarLineas(Mat img, int tipo){
	
	//Tipo: 0 = Vertical
	//		1 = Horizontal
	//		2 = Diagnoal /
	//		3 = Diagonal \
	
	Mat Prewitt1;
	Mat kernel = Mat::eye( 3, 3, CV_32FC1 );
	switch(tipo){
	case 0 : //Vertical
		
		*(((float*) (kernel.data))) = -1;  // [0][0] 
		*(((float*) (kernel.data))+1) = 2;  // [0][1] 
		*(((float*) (kernel.data))+2) = -1.0;  // [0][2] 
		*(((float*) (kernel.data))+3) = -1.0; //[1][0]
		*(((float*) (kernel.data))+4) = 2.0;// [1][1]
		*(((float*) (kernel.data))+5) = -1.0;// [1][2]
		*(((float*) (kernel.data))+6) = -1.0;// [2][0]
		*(((float*) (kernel.data))+7) = 2.0;// [2][1]
		*(((float*) (kernel.data))+8) = -1.0;// [2][2]
		
		filter2D(img, Prewitt1, -1, kernel);
		break;
	case 1 : //Horizontal
		
		
		*(((float*) (kernel.data))) = -1.0;  // [0][0] 
		*(((float*) (kernel.data))+1) = -1.0;  // [0][1] 
		*(((float*) (kernel.data))+2) = -1.0;  // [0][2] 
		*(((float*) (kernel.data))+3) = 2.0; //[1][0]
		*(((float*) (kernel.data))+4) = 2.0;// [1][1]
		*(((float*) (kernel.data))+5) = 2.0;// [1][2]
		*(((float*) (kernel.data))+6) = -1.0;// [2][0]
		*(((float*) (kernel.data))+7) = -1.0;// [2][1]
		*(((float*) (kernel.data))+8) = -1.0;// [2][2]
		
		filter2D(img, Prewitt1, -1, kernel);
		return Prewitt1;
	case 2 : //Diagonal /
		
		
		*(((float*) (kernel.data))) = -1.0;  // [0][0] 
		*(((float*) (kernel.data))+1) = -1.0;  // [0][1] 
		*(((float*) (kernel.data))+2) = 2.0;  // [0][2] 
		*(((float*) (kernel.data))+3) = -1.0; //[1][0]
		*(((float*) (kernel.data))+4) = 2.0;// [1][1]
		*(((float*) (kernel.data))+5) = -1.0;// [1][2]
		*(((float*) (kernel.data))+6) = 2.0;// [2][0]
		*(((float*) (kernel.data))+7) = -1.0;// [2][1]
		*(((float*) (kernel.data))+8) = -1.0;// [2][2]
		
		filter2D(img, Prewitt1, -1, kernel);
		return Prewitt1;
	case 3 : //Diagonal \
		
		
		*(((float*) (kernel.data))) = 2.0;  // [0][0] 
		*(((float*) (kernel.data))+1) = -1.0;  // [0][1] 
		*(((float*) (kernel.data))+2) = -1.0;  // [0][2] 
		*(((float*) (kernel.data))+3) = -1.0; //[1][0]
		*(((float*) (kernel.data))+4) = 2.0;// [1][1]
		*(((float*) (kernel.data))+5) = -1.0;// [1][2]
		*(((float*) (kernel.data))+6) = -1.0;// [2][0]
		*(((float*) (kernel.data))+7) = -1.0;// [2][1]
		*(((float*) (kernel.data))+8) = 2.0;// [2][2]
		
		filter2D(img, Prewitt1, -1, kernel);
		return Prewitt1;
	}


}


Mat Segmentar(Mat imagen, int Hmin, int Hmax,int Smin, int Smax){
	//Variables
	Mat mascara = cv::Mat::zeros(imagen.size(),imagen.type());
	vector<Mat> hsv_channels(3);
	Mat imgHSV;
	
	//Paso de RGB a HSI
	cvtColor(imagen,imgHSV,COLOR_BGR2HSV);
	//Divido los canales
	split(imgHSV,hsv_channels);
	
	
	///Recorro las componentes H y S, entonces si estos valores están dentro de 
	///los minimos y maximos definidos anteriormente a la mascara le coloco 1,
	///sino queda en cero. Entonces en otras palabras, si encuentra el amarillo
	/// a la mascara la dejo en 1, de lo contrario 0.
	for (int i = 0; i<imagen.rows;i++)
		for (int j = 0; j<imagen.cols;j++){
			int H = (int)hsv_channels[0].at<uchar>(i,j);
			int S = (int)hsv_channels[1].at<uchar>(i,j);
			if ((Hmin<H)&&(H<Hmax) && (Smin<S)&&(S<Smax)){
				mascara.at<cv::Vec3b>(i,j)[0]=1; //Blue
				mascara.at<cv::Vec3b>(i,j)[1]=1; //Green
				mascara.at<cv::Vec3b>(i,j)[2]=1; //Red
			}
	}
		
		///Multiplico la imagen con la mascara y asi obtengo solo el cartel
		Mat imgSegmentada;
		multiply(imagen,mascara,imgSegmentada); //Multiplico la imagen por la mascara
		
		return imgSegmentada;
}











Point2d EsquinaSuperiorIzquierda(Mat img){
	Point2d p;
	
	for(int i = 0; i<img.rows;i++){
		for(int j = 0; j<img.cols;j++){
			int intensidad = (int) img.at<uchar>(i,j);
			if(intensidad != 0){
				p.x = i;
				p.y = j;
				return p;
				
			}	
		}
	}
}


Point2d EsquinaInferiorIzquierda(Mat img){
	Point2d p;
	
	for(int i = 0; i<img.rows;i++){
		for(int j = img.cols-1; j>0;j--){
			int intensidad = (int) img.at<uchar>(i,j);
			if(intensidad != 0){
				p.x = i;
				p.y = j;
				return p;
				
			}	
		}
	}
}

Point2d EsquinaSuperiorDerecha(Mat img){
	Point2d p;
	
	for(int i = img.rows-1; i>0;i--){
		for(int j = 0; j<img.cols;j++){
			int intensidad = (int) img.at<uchar>(i,j);
			if(intensidad != 0){
				p.x = i;
				p.y = j;
				return p;
				
			}	
		}
	}
}

Point2d EsquinaInferiorDerecha(Mat img){
	Point2d p;
	
	
	for(int i = img.rows-1; i>0;i--){
		for(int j = img.cols-1; j> 0;j--){
			int intensidad = (int) img.at<uchar>(i,j);
			if(intensidad != 0){
				p.x = i;
				p.y = j;
				return p;
				
			}	
		}
	}
}


Mat Recortar(Mat img){
	
	for(int i = 0; i<img.rows;i++){
		for(int j = 0; j<img.cols;j++){
			int intensidad = (int) img.at<uchar>(i,j);
			if(intensidad != 0){
				Mat Subimagen = img(Rect(j,i,70,130));
				return	Subimagen;
			}	
		}
	}
}


vector<int> AcumuladorSuperior(Mat img){
	vector<int> v(img.cols,0);
	
	for(int i = 0; i<img.cols; i++){
		for(int j = 0; j< (int) img.rows/3;j++){
			int intensidad = (int) img.at<uchar>(j,i) ;
			if(intensidad == 0)
				v[i] +=1;
		}
	}
	return v;
}


vector<int> AcumuladorInferior(Mat img){
	vector<int> v(img.cols,0);
	for(int i = 0; i<img.cols; i++){
		for(int j = img.rows-1; j>(int)(img.rows-img.rows/5); j--){
			int intensidad = (int) img.at<uchar>(j,i) ;
			if(intensidad == 0)
				v[i] +=1;
		}
	}
	return v;
}

int CuentaFlechas(vector<int> v){
	vector<int> n;
	for(int i = 0; i<v.size(); i++){
		int j = i;
		while(i != v.size() && v[i] !=0)
			i = ++i;
		if( i-j != 0)
			n.push_back(1);
		
	}
	return n.size();			
}






	
	
	int main(int argc, char** argv) {
		Mat imagen = imread("iguazu_ruidogris.jpg",IMREAD_GRAYSCALE);
		//	Mat imagen = imread("corrientes_ruidogris.jpg",IMREAD_GRAYSCALE);
		//	Mat imagen = imread("corrientes_ruidogris_0.jpg",IMREAD_GRAYSCALE);
		//	Mat imagen = imread("corrientes_ruidogris_70.jpg",IMREAD_GRAYSCALE);
		imshow("Original",imagen);
		
		///Aplico un filtro de alfa recortado
		filtro_mediaAlfa_recortado(imagen,2,5);
		imshow("Filtrada",imagen);
		
		///Binarizo
		threshold(imagen,imagen,100,255,THRESH_BINARY);
		imshow("Binarizada",imagen);
		
		///3 Erosiones y 1 Apertura
		Mat ee = getStructuringElement(MORPH_RECT,Size(3,3));
		morphologyEx(imagen,imagen,MORPH_ERODE,ee);
		morphologyEx(imagen,imagen,MORPH_ERODE,ee);
		morphologyEx(imagen,imagen,MORPH_ERODE,ee);
		morphologyEx(imagen,imagen,MORPH_OPEN,ee);
		imshow("Apertura",imagen);
		
		///Detecto las Lineas
		imagen = DetectarLineas(imagen,3);
		imshow("Lineas",imagen);
		
		///Aplico Hough
		double nLineas = 50;
		Mat imagenFinal = DibujarHough(imagen,nLineas);
		imshow("Hough",imagenFinal);
		
		
		waitKey(0);
		return 0;
	} 
	
