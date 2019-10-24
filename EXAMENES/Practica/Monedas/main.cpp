#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "pdi_functions.h"

using namespace cv;
using namespace std;
using namespace pdi;

int v1x=-1,v1y=0;
bool bandera=false;


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

void FindBlobs(const cv::Mat &binary, vector < vector<cv::Point>  > &blobs)
{
	blobs.clear();
	
	// Fill the label_image with the blobs
	// 0  - background
	// 1  - unlabelled foreground
	// 2+ - labelled foreground
	
	///input is a binary image therefore values are either 0 or 1
	///out objective is to find a set of 1's that are together and assign 2 to it
	///then look for other 1's, and assign 3 to it....so on a soforth
	
	cv::Mat label_image;
	binary.convertTo(label_image, CV_32FC1); // weird it doesn't support CV_32S! Because the CV::SCALAR is a double value in the function floodfill
	
	int label_count = 2; // starts at 2 because 0,1 are used already
	
	//erode to remove noise-------------------------------
	Mat element = getStructuringElement( MORPH_RECT,
										Size( 2*3 + 1, 2*3+1 ),
										Point( 0, 0 ) );
	/// Apply the erosion operation
	erode( label_image, label_image, element );
	//---------------------------------------------------
	
	//just check the Matrix of label_image to make sure we have 0 and 1 only
	//cout << label_image << endl;
	
	for(int y=0; y < binary.rows; y++) {
		for(int x=0; x < binary.cols; x++) {
			float checker = label_image.at<float>(y,x); //need to look for float and not int as the scalar value is of type double
			cv::Rect rect;
			//cout << "check:" << checker << endl;
			if(checker == 1) {
				//fill region from a point
				cv::floodFill(label_image, cv::Point(x,y), cv::Scalar(label_count), &rect, cv::Scalar(0), cv::Scalar(0), 4);
				label_count++;
				//cout << label_image << endl <<"by checking: " << label_image.at<float>(y,x) <<endl;
				//cout << label_image;
				
				//a vector of all points in a blob
				std::vector<cv::Point> blob;
				
				for(int i=rect.y; i < (rect.y+rect.height); i++) {
					for(int j=rect.x; j < (rect.x+rect.width); j++) {
						float chk = label_image.at<float>(i,j);
						//cout << chk << endl;
						if(chk == label_count-1) {
							blob.push_back(cv::Point(j,i));
						}                        
					}
				}
				//place the points of a single blob in a grouping
				//a vector of vector points
				blobs.push_back(blob);
			}
		}
	}
	cout << label_count <<endl;
info(label_image);
label_image.convertTo(label_image,CV_8UC1);
/*normalize(label_image,label_image,0,255,CV_MINMAX);*/
/*print(label_image,cout);*/
	imshow("label image",label_image);
}

int main(int argc, char** argv) {
	namedWindow("Original",CV_WINDOW_KEEPRATIO);
	Mat img = imread("01_Monedas.jpg");
	int f=img.rows, c=img.cols;
	
	//HSV------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	Mat colorHsv = img.clone();
	cvtColor(colorHsv,colorHsv,COLOR_BGR2HSV);
	
	
	Mat binaria;
	cvtColor(img,binaria,COLOR_RGB2GRAY);
	binaria.convertTo(binaria,CV_8UC1);
	binaria = umbralBinario(binaria,0.95,false);
	imshow("Binaria",binaria);

	
	//	//MORFOLOGIA (en segmentadoColor esta la imagen en blanco y negro 1 canal)------------------------------------------------------------------------------------------------------
	//	Point p1(130,203),p2(175,241); //Rectangulo de colores en los cuales aplicar la esfera para segmentar
	//	Mat binaria = segmentarColor(p1.x,p1.y,p2.x,p2.y,colorHsv);
	//	Mat morf;
	//	Mat ee = getStructuringElement( MORPH_CROSS, Size(9,9));
	//	erode(binaria,morf,ee,Point(-1,-1),10);
	//	imshow("Erosion",binaria);
	//	dilate(binaria,morf,ee,Point(-1,-1),1);
	//	imshow("Morfologia",morf);
	//	//MORFOLOGIA (en segmentadoColor esta la imagen en blanco y negro 1 canal)------------------------------------------------------------------------------------------------------
	
	//CONTORNOS------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	Mat contours = Mat::zeros(f,c,CV_8UC1);
	vector<Vec4i> nosequees;
	vector<vector<Point> > contornos;
	findContours(binaria.clone(),contornos,nosequees, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));
	
	
//	Mat binaria2=binaria.clone();
//	binaria2.convertTo(binaria2, CV_32F, 1./255, 0); 
//	vector < vector<cv::Point>  > blobs;
//	FindBlobs(binaria2, blobs);
	
	
	Mat prueba=binaria.clone();
	int cant=connectedComponents(binaria,prueba,8,CV_32S);
	cout<<cant<<endl;
	prueba.convertTo(prueba,CV_8UC1);
	imshow("prueba",prueba);
	drawContours(contours, contornos, -1, 255,-1,8);
	vector<double> areas(contornos.size());//Area de todos los contornos
	for (int i=0;i<contornos.size();i++){
		double area = contourArea(contornos[i]);
		areas[i]=area;
		//		cout<<area<<endl;
	}
	//	sort(areas.begin(),areas.end());//Ordeno areas
	int cont[8][2];
	cont[0][0]=3150;
	cont[1][0]=4100;
	cont[2][0]=4550;
	cont[3][0]=5300;
	cont[4][0]=5850;
	cont[5][0]=6350;
	cont[6][0]=6950;
	cont[7][0]=7700;
	cont[0][1]=0;
	cont[1][1]=0;
	cont[2][1]=0;
	cont[3][1]=0;
	cont[4][1]=0;
	cont[5][1]=0;
	cont[6][1]=0;
	cont[7][1]=0;
	for (int i=1;i<areas.size();i++){
		double area = areas[i];
		for (int j=0;j<8;j++){
			if(abs(area-cont[j][0])<=50){
				cont[j][1]++;
				break;
			}
		}
		cout<<area<<endl;
	}
	for (int i=0;i<8;i++){
		cout<<"Hay "<<cont[i][1]<<"monedas del tipo "<<i+1<<endl;
	}
	//	int cantidadAreasMayores = 2;//Cantidad de areas mayores con las que me quedo
	//	if(cantidadAreasMayores>areas.size()){
	//		cout<<"HAY MENOS AREAS"<<endl;
	//		cantidadAreasMayores = areas.size();
	//	}
	//	int a=areas.size()-1, b=areas.size()-cantidadAreasMayores-1;
	//	for(int i=a;i>b;i--){
	//		double aref = areas[i];
	//		for (int j=0;j<contornos.size();j++){
	//			double area = contourArea(contornos[j]);
	//			if (area==aref)
	//				drawContours(contours, contornos, j, 255,-1,8);
	//		}
	//	}
	imshow("Contornos",contours);//Contours tiene las areas de mayor contorno
	//CONTORNOS------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	
	//	//HOUGH------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	//	Mat bordes = contours.clone(),lineasHough = img.clone();
	//	Canny(bordes,bordes,250,250);
	//	imshow("Bordes",bordes);
	//	vector<Vec2f> lines;
	//	HoughLines(bordes, lines, 1, CV_PI/180, 20, 0, 0);
	//	int s = lines.size();
	//	float centroRho = 0, centroTheta = 0, tolRho = 200, tolTheta = 0.2; //filtros (para rho y theta) que segmentan las lineas 
	//	for(int i=0;i<s;i++){
	//		float rho = lines[i][0], theta = lines[i][1];
	//		if(abs(rho-centroRho)<tolRho && abs(theta-centroTheta)<tolTheta){
	//			Point pt1, pt2;
	//			double a = cos(theta), b = sin(theta);
	//			double x0 = a*rho, y0 = b*rho;
	//			pt1.x = cvRound(x0 + 1000*(-b));
	//			pt1.y = cvRound(y0 + 1000*(a));
	//			pt2.x = cvRound(x0 - 1000*(-b));
	//			pt2.y = cvRound(y0 - 1000*(a));
	//			line( lineasHough, pt1, pt2, Scalar(0,0,255), 1, CV_AA);
	//		}
	//	}
	//	imshow("Hough",lineasHough); //lineasHough tiene las lineas segmentadas
	//HOUGH------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
	
	imshow("Original",img);
	//	imshow("Original1",img);
	setMouseCallback("Original",valorPixel,&colorHsv);
	//	setMouseCallback("Original1",segmentarColor1,&colorHsv);
	
	waitKey(0);
	return 0;
} 

