#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>


using namespace std;
using namespace cv;


Mat img1;

static void onMouse( int event, int x, int y, int f, void* ){
	
	
	Mat image=img1.clone();
	
	if( event != CV_EVENT_LBUTTONDOWN )
		return;
	
	
	if( event == CV_EVENT_LBUTTONDOWN )
	{
		cout<<"x: "<<x<<"   y: "<<y<<endl;
		
		//para escala de color
		
		cout<<"intensidad rgb: "<<image.at<Vec3b>(y, x)<<endl;
		
		// para escala de gris
		
		//cout<<"intensidad: "<<image.at<uchar>(y,x)<<endl;
		
		//para hsv
		
		Mat image2;
		cvtColor(image,image2,CV_BGR2HSV);
		
		cout<<"intensidad hsv: "<<image2.at<Vec3b>(y, x)<<endl;
	}
	
}


int main(int argc, char *argv[]) {
	
	
	/****************************************/
	// mouse
	
	
	img1 = cv::imread("train1.png");  // crear una imagen desde un archivo
	
		
	cv::namedWindow("v1");  // crea ventana
	cv::imshow("v1", img1);  // asocia imagen con ventana
	cout<<"ancho: "<<img1.size().width<<endl;
	cout<<"alto: "<<img1.size().height<<endl;
	setMouseCallback("v1", onMouse, 0);
	waitKey();
	
	
	
	
	
	//------------------------------------------------------------------------
	
	/****************************************/
	// histograma (Escala de gris)  (hecho a pata)
	
	/*
	
	Mat img = imread("img/estanbul.tif",CV_LOAD_IMAGE_GRAYSCALE);  
	
	// creo el vector
	vector<int> histograma(256);
	
	//ante la duda seteo todo
	for(int i = 0; i < 255; i++)
	{
		histograma.at(i) = 0;
	}
	
	// recorro filas 
	for(int y = 0; y < img.rows; y++)
	{   //recorro columnas 
		for(int x = 0; x < img.cols; x++)
		{
			//cargo el valor de ese pixel (x,y)
			histograma.at((int)img.at<uchar>(x,y))++;
		}
	}
	
	//muestro por pantalla valores 
	
	
	//for(int i = 0; i < 256; i++)
	//cout<<histograma.at(i)<<" "<<endl;
	
	
	// dibujo
	
	//tamanio del histograma
	int hist_w = 512; 
	int hist_h = 400;
	
	
	//REFERENCIA
	//histm genera el histograma
	//hist_h 
	//hist_w
	//CV_8UC1 escala de grises
	//scalar(255,255,255) 
	
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255, 255, 255));
	
	// find the maximum intensity element from histogram
	int max = histograma.at(0);
	for(int i = 1; i < 256; i++){
		if(max < histograma[i]){
			max = histograma[i];
		}
	}
	
	//normalizo en e rango 0 -> maximo
	
	for(int i = 0; i < 255; i++)
	{
		histograma[i] = ((double)histograma[i]/max)*histImage.rows;
	}
	
	int bin_w = cvRound((double) hist_w/256);
	
	// dibujo el histgrama
	for(int i = 0; i < 255; i++)
	{
		
		//revisar esta linea copypast
		line(histImage, Point(bin_w*(i), hist_h),
			 Point(bin_w*(i), hist_h - histograma[i]),
			 Scalar(0,0,0), 1, 8, 0);
	}
	
	// dibujo el histograma
	namedWindow("Histograma", CV_WINDOW_AUTOSIZE);
	imshow("Histograma", histImage);
	
	waitKey();
	
	
		
	
	
	*/
	
	/****************************************/
	// histograma (color) (hecho a pata)
	
	/*
	Mat image = imread("img/goku.jpg");  
	
	
	int HistR[257] = {0};
	int HistG[257] = {0};
	int HistB[257] = {0};
	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
	{
			Vec3b intensity = image.at<Vec3b>(Point(j, i));
			int Red = intensity.val[0];
			int Green = intensity.val[1];
			int Blue = intensity.val[2];
			HistR[Red] = HistR[Red]+1;
			HistB[Blue] = HistB[Blue]+1;
			HistG[Green] = HistG[Green]+1;
	}
		Mat HistPlotR (1000, 256, CV_8UC3, Scalar(0, 0, 0));
		Mat HistPlotG (1000, 256, CV_8UC3, Scalar(0, 0, 0));
		Mat HistPlotB (1000, 256, CV_8UC3, Scalar(0, 0, 0));
		for (int i = 0; i < 256; i=i+2)
		{
			line(HistPlotR, Point(i, 1000), Point(i, 1000-HistR[i]), Scalar(0, 0, 255),1,8,0);
			line(HistPlotG, Point(i, 1000), Point(i, 1000-HistG[i]), Scalar(0, 255, 0),1,8,0);
			line(HistPlotB, Point(i, 1000), Point(i, 1000-HistB[i]), Scalar(255, 0, 0),1,8,0);
		}
		
		namedWindow("Red Histogram");
		namedWindow("Green Histogram");
		namedWindow("Blue Histogram");
		namedWindow("ima");
		imshow("ima", image);
		waitKey();
		imshow("Red Histogram", HistPlotR);
		imshow("Green Histogram", HistPlotG);
		imshow("Blue Histogram", HistPlotB);
		waitKey();
	
		
	
		
		*/
	
	/****************************************/
	// histograma escala de grises  (automatico) 
	// Tambien se puede aplicar a color sin problema
	
	
	/*
	Mat img = imread("img/estanbul.tif",CV_LOAD_IMAGE_GRAYSCALE);  
	
	MatND hist;
	
	int bins=256;
	float lranges[] = {0, 256};
	int channels[] = {0};
	int histSize[] = {bins};
	const float* ranges[] = {lranges};
	int numimgs = 1; 
	int dims=1;
	
	calcHist(
			 &img, // imagen de entrada
			 numimgs,  //cantidad de imagenes
			 channels, //cantidad canales
			 Mat(), // mascara o pixeles a considerar (aca no usa ninguna)
			 hist,  // imagen de salida
			 dims,   //dimensiones (unidimensional en este caso)
			 histSize,   // cantidad de cubetas o niveles
			 ranges,  // rangos o valores limites
			 true, //  histograma uniforme
			 false  // acumulador
			 ); 
	
	Mat histNorm = hist / (img.rows * img.cols);  // histograma normalizado
	
	
	// create matrix for histogram visualization
	int const hist_height = 256;
	cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);
	
	double max_val=0;
	minMaxLoc(hist, 0, &max_val);
	
	// visualize each bin
	for(int b = 0; b < bins; b++) {
		float const binVal =  hist.at<float>(b);
		int   const height = cvRound(binVal*hist_height/max_val);
		cv::line
			( hist_image
			 , cv::Point(b, hist_height-height), cv::Point(b, hist_height)
			 , cv::Scalar::all(255)
			 );
	}
	
	
	namedWindow("Image",CV_WINDOW_AUTOSIZE);
	imshow("Image",img);
	waitKey();
	namedWindow("Histograma",CV_WINDOW_AUTOSIZE);
	imshow("Histograma", hist_image);
	waitKey();
	
	*/
		
		
	
	
	
    //------------------------------------------------------------------------
	
	
	
	/********************************************************/
	// perfil de intensidades en cada valor d un punto en una columna o fila
	
	/*
	
	Mat img = imread("img/riuk.jpg",CV_LOAD_IMAGE_GRAYSCALE);  
	
	//columna elegida
	int y=50;
	int q=0;
	
	Mat inten(255,img.cols, CV_8UC3, Scalar(0,0,0));
	
	//Mat (rows (y), cols(x), 8bits Cantidad canales, valor inicial cada pixel)
	
	Vec3b jed;
	jed[0]=255;
	jed[1]=255;
	jed[2]=255;
	
	
	
	//lo calculo asi y despues lo roto a 180°
	
	for (int i=0;i<img.cols;i++){
	q=img.at<uchar>(y,i);
	cout<<" intensidad en punto: "<<q<<endl; 
	inten.at<Vec3b>(q,i)=jed;  
	for (int d=0;d<q;d++){  
    inten.at<Vec3b>(d,i)=jed;
	}
	}
	
	//roto la Imagen
	
	double angle = 180;
	
	// get rotation matrix for rotating the image around its center
	cv::Point2f center(inten.cols/2.0, inten.rows/2.0);
	cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
	
	// determine bounding rectangle
	cv::Rect bbox = cv::RotatedRect(center,inten.size(), angle).boundingRect();
	
	// adjust transformation matrix
	rot.at<double>(0,2) += bbox.width/2.0 - center.x;
	rot.at<double>(1,2) += bbox.height/2.0 - center.y;
	
	cv::Mat dst;
	cv::warpAffine(inten, dst, rot, bbox.size());
	
	
	
	//pongo eje normal (y abajo) y lo calculo
	
	Mat inten2(255,img.cols, CV_8UC3, Scalar(0,0,0));
	
	
	for (int i=0;i<img.cols;i++){
		q=img.at<uchar>(y,i);
		inten2.at<Vec3b>(254-q,i)=jed;  
		for (int d=254-q;d<255;d++){  
			inten2.at<Vec3b>(d,i)=jed;
		}
	}
	
	
	namedWindow("Image");
	imshow("Image",img);
	waitKey();
	namedWindow("Intensidad rotada");
	imshow("Intensidad rotada", dst);
	waitKey();
	namedWindow("Intensidad");
	imshow("Intensidad", inten2);
	waitKey();
	
	waitKey();

	
	*/
	

	
	
	//------------------------------------------------------------------------
	
	
	/****************************************/
	
	//perfil de intensidad de un segmento de la Imagen
	
	// Hay q ver a q se refiere con segmento
	
	//corto una parte de la imagen (Region of interest)
	
	//cv::Rect roi(0, 0, 200, 220);  //pos x, pos y, ancho, alto
	//cv::Mat image_roi = img1(roi);
	
	
	
	
	
	return 0;
}	
	
	
	
	
	


