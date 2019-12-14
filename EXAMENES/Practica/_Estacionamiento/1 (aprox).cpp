#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
	
	//cargo imagen
	
	cv::Mat img1 = cv::imread("1.png");  // crear una imagen desde un archivo
	
	cv::imshow("Original", img1);  // asocia imagen con ventana
	waitKey();
	
	cout<<"columnas: "<<img1.cols<<endl; // lo mismo que ancho
	cout<<"filas: "<<img1.rows<<endl;  // lo mismo que alto
	
	

	int auto_posx = 0;
	int auto_posy = 0;
	
	int linea_p1sx=0;
	int linea_p1sy=0;
	
	int linea_p1ix=0;
	int linea_p1iy=0;
	
	int linea_p2sx=0;
	int linea_p2sy=0;
	
	int linea_p2ix=0;
	int linea_p2iy=0;	

	
	int com=-1;

	
	Vec3b jerk;  // tiene 3 espacios. Vec3f es para flotantes
	jerk.val[0]=196;
	jerk.val[1]=196;
	jerk.val[2]=196;
	
	
	Vec3b auti;  // tiene 3 espacios. Vec3f es para flotantes
	auti.val[0]=161;
	auti.val[1]=109;
	auti.val[2]=208;
	
	
	
	for (int i=0;i<img1.rows;i++){
		for (int j=0;j<img1.cols;j++){
	 
	if ( (img1.at<Vec3b>(i,j)==jerk) && (com==-1) )
		{
		  com=1;
		  linea_p1sx=j;
		  linea_p1sy=i;
		  
		  int aux=linea_p1sy;
			  
		  while (true){
			  if (img1.at<Vec3b>(aux,j)!=jerk){
				  linea_p1ix=linea_p1sx;
				  linea_p1iy=aux;
				  break;
			  }

			  aux++;
		  }
		  
		  
		  aux=linea_p1sx;
		  
		  while (true){
			  if (img1.at<Vec3b>(i,aux)!=jerk){
			      j=aux;
				  break;
			  }
			  
			  aux++;
			  
		}
	
	}
	
	if ( (img1.at<Vec3b>(i,j)==jerk) && (com==1) )
	{
		com=2;
		linea_p2sx=j;
		linea_p2sy=i;
		
		int aux=linea_p2sy;
		
		while (true){
			if (img1.at<Vec3b>(aux,j)!=jerk){
				linea_p2ix=linea_p1sx;
				linea_p2iy=aux;
				break;
			}
			aux++;
		}
		
	}
	
	
	
	if  (img1.at<Vec3b>(i,j)==auti){
		    auto_posx=j;
			auto_posy=i;
	}
	
	
	
	
	 }
	}
		
		
		
		int ct_1y= (linea_p1iy+linea_p1sy)/2;
		int ct_2y= (linea_p2iy+linea_p2sy)/2;
		
		int centroy = (ct_2y + ct_1y)/2;

		int ct_1x = (linea_p1ix+linea_p1sx)/2;
		int ct_2x= (linea_p2ix+linea_p2sx)/2;
		
		int centrox = (ct_2x + ct_1x)/2;
	
		
		
		
		Vec3b marcar;  // tiene 3 espacios. Vec3f es para flotantes
		marcar.val[0]=0;
		marcar.val[1]=0;
		marcar.val[2]=250;
		
		
		
		if (auto_posx>centrox)  {     //auto esta a la derecha de estacionamiento
			
			if (auto_posy>centroy){    //auto esta abajo del estacionamiento
				
					for (int j=auto_posx;j>centrox;j--){
						img1.at<Vec3b>(auto_posy,j)=marcar;
					}
					
					for (int i=auto_posy;i>centroy;i--){
						img1.at<Vec3b>(i,auto_posx)=marcar;
					}		
			}
			
			if (auto_posy<centroy){    //auto esta arriba del estacionamiento
				
				for (int j=auto_posx;j>centrox;j--){
					img1.at<Vec3b>(auto_posy,j)=marcar;
				}
				
				for (int i=auto_posy;i<centroy;i++){
					img1.at<Vec3b>(i,auto_posx)=marcar;
				}
				
		}
		
		}
		
		
		
		
		if (auto_posx<centrox) {  //auto esta a la izquierda de estacionamiento
		
			if (auto_posy>centroy){    //auto esta abajo del estacionamiento
				
				for (int j=auto_posx;j<centrox;j++){
					img1.at<Vec3b>(auto_posy,j)=marcar;
				}
				
				for (int i=auto_posy;i>centroy;i--){
					img1.at<Vec3b>(i,auto_posx)=marcar;
				}		
			}
			
			if (auto_posy<centroy){    //auto esta arriba del estacionamiento
				
				for (int j=auto_posx;j<centrox;j++){
					img1.at<Vec3b>(auto_posy,j)=marcar;
				}
				
				for (int i=auto_posy;i<centroy;i++){
					img1.at<Vec3b>(i,auto_posx)=marcar;
				}			
			}
		}
			
		
		cv::imshow("Estaciona", img1);  // asocia imagen con ventana
		waitKey();
		
		
	return 0;
}

