/*
* PDI_functions
*/
#ifndef PDI_FUNCTIONS_H
#define PDI_FUNCTIONS_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>




/**Funciones auxiliares
*/
namespace pdi{
	typedef unsigned char byte;
	
	/**Modifica la matriz para que todos los valores queden dentro del rango,
	los valores fuera del rango se setean al extremo más cercano.
	*/
	void clamp(cv::Mat& mat, double lowerBound, double upperBound);
	
	/**Imprime información de tipo y tamaño.
	*/
	void info(const cv::Mat &m, std::ostream &out = std::cout);
	
	/**Imprime valores estadísticos: mínimo, máximo, media, desvío.
	BUG: max, min, funcionan sólo con imágenes de un canal.
	*/
	void stats(const cv::Mat &m, std::ostream &out = std::cout);
	
	/**Muestra en pantalla la matriz
	*/
	void print(const cv::Mat &image, std::ostream &out = std::cout);
	
	/**Realiza el intercambio al copiar elemento a elemento
	Deben ser matrices de igual tamaño
	*/
	void swap_copy(cv::Mat &a, cv::Mat &b);
	
	/**Desplaza la imagen de modo que el píxel central
	ocupe la esquina superior izquierda.
	Usado para visualizar la transformada de Fourier.
	*/
	void centre(cv::Mat &imagen);
	
	/** Histograma uniforme de una imagen de un canal de 8bits.
	\param mask permite seleccionar una región a procesar
	*/
	cv::Mat histogram(const cv::Mat &image, int levels, const cv::Mat &mask=cv::Mat());
	
	/**Dibuja un gráfico de líneas en el canvas.
	* \param data vector con los valores a graficar,
	* rango [0,1] para flotantess o [0,MAX] para enteros, que se mapean del borde inferior al superior.
	* \param color, color de las lí
	*/
	cv::Mat draw_graph(
					   const cv::Mat &data,
					   cv::Scalar color = cv::Scalar::all(255)
					   );
	
	/**Devuelve un gráfico de líneas comparativo.
	*/
	cv::Mat draw_graph(
					   const std::vector<cv::Mat> &data,
					   const std::vector<cv::Scalar> colour
					   );
	
	
	/**Copia la imagen a una cuyas dimensiones hacen eficiente la fft
	*/
	cv::Mat optimum_size(const cv::Mat &image);
	
	/**Devuelve la magnitud logarítmica del espectro, centrada.
	Úsese para visualización.
	\param image debe ser de 32F
	*/
	cv::Mat spectrum(const cv::Mat &image);
	
	/**Devuelve la convolución.
	Kernel centrado, borde constante.
	*/
	cv::Mat convolve(const cv::Mat &image, const cv::Mat &kernel);
	
	
	/**Concatena las imágenes.
	* Los tamaños deberían concordar.
	*/
	cv::Mat mosaic( const cv::Mat &a, const cv::Mat &b, bool vertical=true );
	
	/**Concatena las imágenes en el vector
	* Los tamaños deberían concordar.
	\param r: número de filas
	*/
	cv::Mat mosaic( const std::vector<cv::Mat> &images, size_t r=1);
	
	/**Devuelve los mapeos para rgb.
	* Las matrices son de tipo 8U
	*/
	std::vector<cv::Mat> load_colormap(const char *filename);
	
	/**Rota la imagen en sentido antihorario
	\param angle en grados
	*/
	cv::Mat rotate(cv::Mat image, double angle);
	
	/**Realiza el filtrado en frecuencia
	\param image matriz 32F, un canal.
	\param filtro de magnitud, descentrado.
	*/
	cv::Mat filter(cv::Mat image, cv::Mat filtro);
	
	
	/**Devuelve una imagen descentrada de la magnitud de un filtro ideal
	\param corte frecuencia de corte relativa. 0.5 corresponde un círculo inscripto
	*/
	cv::Mat filter_ideal(size_t rows, size_t cols, double corte);
	
	/**Devuelve una imagen descentrada de la magnitud de un filtro butterworth
	*/
	cv::Mat filter_butterworth(size_t rows, size_t cols, double corte, size_t order);
	
	/**Devuelve una imagen descentrada de la magnitud de un filtro gaussiano
	*/
	cv::Mat filter_gaussian(size_t rows, size_t cols, double corte);
	
	
	/**Devuelve una matriz compleja dada su magnitud y fase
	*/
	cv::Mat polar_combine(const cv::Mat &magnitud, const cv::Mat &phase);
	
	/**Dada una matriz compleja, devuelve su magnitud
	*/
	cv::Mat magnitude(const cv::Mat &image);
	
	/**Dada una matriz compleja, devuelve su fase
	*/
	cv::Mat phase(const cv::Mat &image);
	
	/**Devuelve un filtro de desenfoque dadas las velocidades de desplazamiento horizontal y vertical
	*/
	cv::Mat motion_blur(cv::Size size, double a, double b);
}


//Implementation
#include <fstream>
#include <limits>

using namespace cv;
using namespace std;

namespace pdi{
	inline void clamp(cv::Mat& mat, double lowerBound, double upperBound){
		cv::min(cv::max(mat, lowerBound), upperBound, mat);
	}
	
	inline void info(const cv::Mat &image, std::ostream &out){
		out << "Characteristics\n";
		out << "\tSize " << image.rows << 'x' << image.cols << '\n';
		out << "\tChannels " << image.channels() << '\n';
		out << "\tDepth ";
		out << '\t';
		switch(image.depth()){
		case CV_8U: out << "8-bit unsigned integers ( 0..255 )\n"; break;
		case CV_8S: out << "8-bit signed integers ( -128..127 )\n"; break;
		case CV_16U: out << "16-bit unsigned integers ( 0..65535 )\n"; break;
		case CV_16S: out << "16-bit signed integers ( -32768..32767 )\n"; break;
		case CV_32S: out << "32-bit signed integers ( -2147483648..2147483647 )\n"; break;
		case CV_32F: out << "32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )\n"; break;
		case CV_64F: out << "64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )\n"; break;
		}
	}
	
	inline void stats(const cv::Mat &image, std::ostream &out){
		double max, min;
		cv::Mat mean, std;
		cv::meanStdDev(image, mean, std);
		cv::minMaxLoc(image, &min, &max);
		out << "Stats\n";
		out << "\tarea " << image.size().area() << '\n';
		out << "\tminimum " << min << '\n';
		out << "\tmaximum " << max << '\n';
		out << "\tmean " << mean << '\n';
		out << "\tstd " << std << '\n';
	}
	
	inline void print(const cv::Mat &image, std::ostream &out){
		out << image;
	}
	
	inline void swap_copy(cv::Mat &a, cv::Mat &b){
		cv::Mat temp;
		a.copyTo(temp); //NO puede ser reemplazado por .clone()
		b.copyTo(a);
		temp.copyTo(b);
	}
	
	inline void centre(cv::Mat &image){
		int
			cx = image.cols/2,
			cy = image.rows/2,
			w = cx,
			h = cy;
		int
			off_x = image.cols%2,
			off_y = image.rows%2;
		//cuadrantes
		cv::Mat
			top_left = cv::Mat(
							   image,
							   cv::Rect(
										0, 0,
										w, h
										)
							   ),
			top_right = cv::Mat(
								image,
								cv::Rect(
										 cx+off_x, 0,
										 w, h
										 )
								),
			bottom_left = cv::Mat(
								  image,
								  cv::Rect(
										   0, cy+off_y,
										   w, h
										   )
								  ),
			bottom_right = cv::Mat(
								   image,
								   cv::Rect(
											cx+off_x, cy+off_y,
											w, h
											)
								   );
		
		//intercambia los cuadrantes
		swap_copy(top_left, bottom_right);
		swap_copy(top_right, bottom_left);
	}
	
	inline cv::Mat histogram(const cv::Mat &image, int levels, const cv::Mat &mask){
		const int channels = 0;
		float range[] = {0, 256};
		const float *ranges[] = {range};
		
		cv::MatND hist;
		cv::calcHist(
					 &image, //input
					 1, //sólo una imagen
					 &channels, //de sólo un canal
					 mask, //píxeles a considerar
					 hist, //output
					 1, //unidimensional
					 &levels, //cantidad de cubetas
					 ranges //valores límite
					 );
		return hist;
	}
	
	namespace{
		/**Dibuja un gráfico de líneas en el canvas. Función auxiliar
		* \param data vector con los valores a graficar,
		* rango [0,1] para flotantess o [0,MAX] para enteros, que se mapean del borde inferior al superior.
		*/
		cv::Mat draw_graph(
						   cv::Mat &canvas,
						   const cv::Mat &data,
						   cv::Scalar color = cv::Scalar::all(255),
						   double scale = 1
						   );
		
		/**Dibuja un gráfico de líneas en el canvas.
		* wrapper para aceptar std::vector
		*/
		template<class T>
		cv::Mat draw_graph(cv::Mat &canvas, const std::vector<T> &data);
	}
	
	inline
		cv::Mat draw_graph(
						   const std::vector<cv::Mat> &data,
						   const std::vector<cv::Scalar> colour
						   ){
		cv::Mat canvas = cv::Mat::zeros(256, 256, CV_8UC(3) );
		//encontrar el máximo valor entre todos los arreglos
		double max = -std::numeric_limits<double>::max();
		for(size_t K=0; K<data.size(); ++K){
			double max_K;
			cv::minMaxLoc(data[K], NULL, &max_K);
			max = std::max(max, max_K);
		}
		
		//graficar
		for(size_t K=0; K<data.size(); ++K)
			draw_graph(canvas, data[K], colour[K], 1/max);
		
		return canvas;
	}
	
	
	namespace {
		inline cv::Mat draw_graph(cv::Mat &canvas, const cv::Mat &data_, cv::Scalar color, double scale){
			cv::Mat data = data_;
			switch(data_.depth()){
			case CV_8U: data.convertTo(data, CV_32F, 1./255, 0); break;
			case CV_8S: data.convertTo(data, CV_32F, 1./255, 0.5); break;
			case CV_16U: data.convertTo(data, CV_32F, 1./65535, 0); break;
			case CV_16S: data.convertTo(data, CV_32F, 1./65535, 0.5); break;
			case CV_32S: data.convertTo(data, CV_32F, 1./(2*2147483647u+1), 0.5); break;
			case CV_32F:
				break;
			case CV_64F:
				data.convertTo(data, CV_32F, 1);
				break;
			}
			
			double stretch = double(canvas.cols-1)/(std::max(data.rows, data.cols)-1);
			for(int K=1; K<std::max(data.rows, data.cols); ++K){
				cv::line(
						 canvas,
						 cv::Point( (K-1)*stretch, canvas.rows*(1-scale*data.at<float>(K-1)) ),
						 cv::Point( K*stretch, canvas.rows*(1-scale*data.at<float>(K)) ),
						 color
						 );
			}
			
			return canvas;
		}
		
		template<class T>
		inline cv::Mat draw_graph(cv::Mat &canvas, const std::vector<T> &data){
			return draw_graph(canvas, cv::Mat(data));
		}
		
	}
	
	inline cv::Mat draw_graph(const cv::Mat &data, cv::Scalar color){
		cv::Mat canvas = cv::Mat::zeros(256, 256, CV_8UC(3) );
		double max;
		cv::minMaxLoc(data, NULL, &max);
		return draw_graph(canvas, data, color, 1/max);
	}
	
	inline cv::Mat optimum_size(const cv::Mat &image){
		cv::Mat bigger;
		cv::copyMakeBorder(
						   image,
						   bigger,
						   0, cv::getOptimalDFTSize(image.rows)-image.rows,
						   0, cv::getOptimalDFTSize(image.cols)-image.cols,
						   cv::BORDER_CONSTANT
						   );
		
		return bigger;
	}
	
	inline cv::Mat spectrum(const cv::Mat &image){
		cv::Mat fourier;
		cv::dft(image, fourier, cv::DFT_COMPLEX_OUTPUT);
		
		//calcula la magnitud
		std::vector<cv::Mat> xy;
		cv::Mat magnitud;
		cv::split(fourier, xy);
		cv::magnitude(xy[0], xy[1], magnitud);
		
		//logaritmo
		cv::log(magnitud+1, magnitud);
		cv::normalize(magnitud, magnitud, 0, 1, cv::NORM_MINMAX);
		
		//centrado
		centre(magnitud);
		
		return magnitud;
	}
	
	inline cv::Mat convolve(const cv::Mat &image, const cv::Mat &kernel){
		cv::Mat result;
		//same bits as the image, kernel centered, no offset
		cv::filter2D(image, result, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
		return result;
	}
	
	inline cv::Mat mosaic( const cv::Mat &a, const cv::Mat &b, bool vertical ){
		cv::Mat big;
		if(vertical)
			cv::vconcat(a, b, big); //sin documentación
		else
			cv::hconcat(a, b, big); //sin documentación
		
		return big;
	}
	
	inline cv::Mat mosaic( const std::vector<cv::Mat> &images, size_t r){
		if(images.empty()) return cv::Mat();
		size_t c = images.size()/r + ((images.size()%r)?1:0);
		
		size_t rows = images[0].rows, cols = images[0].cols;
		cv::Mat big = cv::Mat::zeros(r*rows, c*cols, images[0].type()); //tamaño total
		
		for(size_t K=0; K<images.size(); ++K){
			cv::Rect submatrix ( (K%c)*cols, (K/c)*rows, images[K].cols, images[K].rows );
			cv::Mat region = cv::Mat(big, submatrix); //región donde pegar
			images[K].copyTo(region);
		}
		
		return big;
	}
	
	inline std::vector<cv::Mat> load_colormap(const char *filename){
		std::ifstream input(filename);
		const size_t size = 256;
		
		std::vector<cv::Mat> rgb(3);
		for(size_t K=0; K<rgb.size(); ++K)
			rgb[K] = cv::Mat::zeros(1, size, CV_8U);
		
		for(size_t K=0; K<size; ++K)
			for(size_t L=0; L<3; ++L){
			float color;
			input>>color;
			rgb[L].at<byte>(K) = 0xff*color;
		}
		
		return rgb;
	}
	
	inline cv::Mat rotate(cv::Mat src, double angle){
		cv::Mat dst;
		cv::Point2f pt(src.cols/2., src.rows/2.);
		cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
		cv::warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
		return dst;
	}
	
	inline cv::Mat filter(cv::Mat image, cv::Mat filtro_magnitud){
		//se asume imágenes de 32F y un canal, con tamaño óptimo
		cv::Mat transformada;
		
		//como la fase es 0 la conversión de polar a cartesiano es directa (magnitud->x, fase->y)
		cv::Mat x[2];
		x[0] = filtro_magnitud.clone();
		x[1] = cv::Mat::zeros(filtro_magnitud.size(), CV_32F);
		
		cv::Mat filtro;
		cv::merge(x, 2, filtro);
		
		cv::dft(image, transformada, cv::DFT_COMPLEX_OUTPUT);
		cv::mulSpectrums(transformada, filtro, transformada, cv::DFT_ROWS);
		
		cv::Mat result;
		cv::idft(transformada, result, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
		return result;
	}
	
	namespace{
		template <class T>
		inline T square(T x){
			return x*x;
		}
		double distance2( double x1, double y1, double x2, double y2 ){
			return square(x2-x1) + square(y2-y1);
		}
	}
	
	inline cv::Mat filter_ideal(size_t rows, size_t cols, double corte){
		cv::Mat
			magnitud = cv::Mat::zeros(rows, cols, CV_32F);
		if(cols%2==1 and rows%2==1) //impar, el centro cae en un píxel
			cv::circle(
					   magnitud,
					   cv::Point(cols/2,rows/2), //punto central
					   rows*corte, //radio
					   cv::Scalar::all(1),
					   -1 //círculo relleno
					   );
		else{
			double limit = square(corte*rows);
			for(size_t K=0; K<rows; ++K)
				for(size_t L=0; L<cols; ++L){
				double d2 = distance2(K+.5, L+.5, rows/2, cols/2);
				if(d2 <= limit)
					magnitud.at<float>(K,L) = 1;
			}
		}
		
		centre(magnitud);
		return magnitud;
	}
	
	inline cv::Mat filter_butterworth(size_t rows, size_t cols, double corte, size_t order){
		//corte = w en imagen de lado 1
		//1 \over 1 + {D \over w}^{2n}
		cv::Mat
			magnitud = cv::Mat::zeros(rows, cols, CV_32F);
		
		corte *= rows;
		//corte *= corte;
		for(size_t K=0; K<rows; ++K)
			for(size_t L=0; L<cols; ++L){
			double d2 = distance2(K+.5, L+.5, rows/2., cols/2.);
			magnitud.at<float>(K,L) = 1.0/(1 + std::pow(d2/(corte*corte), order) );
		}
		
		centre(magnitud);
		return magnitud;
	}
	
	inline cv::Mat filter_gaussian(size_t rows, size_t cols, double corte){
		//corte es sigma en imagen de lado 1
		cv::Mat
			magnitud = cv::Mat::zeros(rows, cols, CV_32F);
		
		corte *= rows;
		//corte *= corte;
		for(size_t K=0; K<rows; ++K)
			for(size_t L=0; L<cols; ++L){
			double distance = distance2(K+.5, L+.5, rows/2., cols/2.);
			magnitud.at<float>(K,L) = std::exp(-distance/(2*corte*corte));
		}
		
		centre(magnitud);
		return magnitud;
	}
	
	inline cv::Mat polar_combine(const cv::Mat &magnitud, const cv::Mat &phase){
		cv::Mat x[2], result;
		cv::polarToCart(magnitud, phase, x[0], x[1]);
		cv::merge(x, 2, result);
		return result;
	}
	
	inline
		cv::Mat magnitude(const cv::Mat &image){
		cv::Mat planes[2];
		cv::split(image, planes);
		
		cv::Mat result;
		cv::magnitude(planes[0], planes[1], result);
		return result;
	}
	
	inline
		cv::Mat phase(const cv::Mat &image){
		cv::Mat phase, planes[2];
		cv::split(image, planes);
		cv::phase(planes[0], planes[1], phase);
		
		return phase;
	}
	
//	a = velocidad en x
//	b = velocidad en y
	inline
		cv::Mat motion_blur(cv::Size size, double a, double b){
		cv::Mat transformation =
			cv::Mat::zeros(size, CV_32FC(2));
		
		int
			rows = transformation.rows,
			cols = transformation.cols;
		
		//const std::complex<float> I{0,1};
		const std::complex<float> I(0,1);
		//fase exp(j\pi (ua + vb))
		//magnitud \frac{ \sin(\pi(ua+vb)) }{ \pi (ua+vb) }
		for(int K=0; K<rows; ++K)
			for(int L=0; L<cols; ++L){
			double
				u = (L-cols/2)/(double)cols,
				v = (K-rows/2)/(double)rows;
			
			float pi_v = M_PI*(u*a+v*b);
			float mag = (pi_v)? sin(pi_v)/pi_v: 1; //lim{x->0} sin(x)/x
			transformation.at< std::complex<float> >(K,L) = mag*exp(I*pi_v);
		}
		
		centre(transformation);
		return transformation;
		}
	
		
	
	
	//----------------------FUNCIONES MIAS-------------------------------------
	//UMBRAL BINARIO
	Mat umbralBinario(Mat img, float u,bool color){//color (del objeto) = false -> pinta blanco el fondo y  negro el objeto
		bool convertir = false;
		switch(img.depth()){
			case CV_8U: img.convertTo(img, CV_32F, 1./255, 0); 
			convertir = true;
			break;
		}
		int f=img.rows, c=img.cols;
		Mat retorno = img.clone();
		float fondo = 0,objeto = 1;
		if (!color){
			fondo = 1;
			objeto = 0;
		}
		
		for (int i=0;i<f;i++){
			for (int j=0;j<c;j++){
				float intensidad = retorno.at<float>(i,j);
				if(intensidad < u)
					intensidad = fondo;
				else
					intensidad = objeto;
				retorno.at<float>(i,j) = intensidad;
			}
		}
		
		if (convertir)
			retorno.convertTo(retorno,CV_8UC1,255);
		
		return retorno;
	}
	//"PRODUCTO PUNTO" ENTRE 2 MATRICES
	float producto(Mat img1, Mat img2){
		int f=img1.rows, c=img1.cols;
		float retorno = 0;
		if (img2.rows==f && img2.cols==c){
			for(int i=0;i<f;i++){
				for(int j=0;j<c;j++){
					float int1=img1.at<float>(i,j),
						int2=img2.at<float>(i,j);
					retorno += (int1*int2);
				}
			}
		}
		retorno /= (f*c);
		return retorno;
	}
	//PARA APLICAR UN FILTRO ESPACIAL
	Mat convolucion(Mat img, Mat kernel){
		info(img);
		int f=img.rows, c=img.cols,
			m=kernel.cols, n=kernel.rows,
			a=m/2, b=n/2;
		Mat aux =  Mat::zeros(f,c,CV_32F), 
			relleno = Mat::zeros(f+2*b,c+2*a,CV_32F);
//		info(img);info(relleno);
		
//		info(img);
		
		img.clone().copyTo(relleno(Rect(a,b,c,f)));
		
//		imshow("Borrar1",relleno);
		
		for(int i=0;i<f-m;i++){
			for(int j=0;j<c-n;j++){
				aux.at<float>(i,j) = producto(img(Rect(j,i,m,n)),kernel);
			}
		}
		//imshow("Borrar2",aux);
		normalize(aux,aux,0,1,CV_MINMAX);
		//	imshow("Borrar3",aux);
		return aux;
	}
	
	
	Mat histograma(Mat img, int cubetas){
		// Solo recibe UCHAR
		img.convertTo(img,CV_8UC1);
		Mat hist = histogram(img,cubetas);
		normalize(hist,hist,0,1,CV_MINMAX);
		Mat hist_img = Mat::zeros(256,256,CV_8UC1);
		draw_graph(hist_img,hist);
		return hist_img;
	}
	
	//ERROR CUADRATICO MEDIO ENTRE 2 MATRICES
	float ecm(Mat img11,Mat img22){
		Mat img1 = img11.clone(), img2 = img22.clone();
		
		switch(img11.depth()){
			case CV_8U: img1.convertTo(img1,CV_32F,1./125);
			normalize(img1,img1,0,1,CV_MINMAX);
			break;
		}
		
		switch(img22.depth()){
		case CV_8U: img2.convertTo(img2,CV_32F,1./125);
		normalize(img2,img2,0,1,CV_MINMAX);
		break;
		}
		
		float retorno = 0;		
		
		int f=img1.rows,c=img1.cols;
		if (img2.rows==f && img2.cols==c){
			
			for (int i=0;i<img1.rows;i++){
				for(int j=0;j<img1.cols;j++){
					float i1 = img1.at<float>(i,j), i2 = img2.at<float>(i,j);
					float el = (float) pow(i1-i2,2);
					retorno += el;
				}
			}
		}
		else
			cout<<"Las dimensiones de las matrices no coinciden\n";
		
		retorno = (float) retorno/(f*c);
		return retorno;
	}
	
	
	//COLOCA LAS IMAGENES DE UN VECTOR EN UNA SOLA
	Mat armar(vector<Mat> imagenes,int columnas){
		int nImagenes = imagenes.size(), f=0,c=0;
		int filas = nImagenes/columnas;
		
		for (int w=0;w<nImagenes;w++){
			Mat img = imagenes[w].clone();
			if (img.rows>f)
				f = img.rows;
			if (img.cols>c)
				c = img.cols;
		}
		
		//Mat retorno = Mat::zeros(filas*f,columnas*c,CV_8UC1);
		Mat retorno = Mat::zeros(filas*f,columnas*c,imagenes[0].depth());
		
		for (int i=0;i<filas;i++){
			for (int j=0;j<columnas;j++){
				//Mat img = imagenes[filas*i+j];
				Mat img = imagenes[i*columnas+j].clone();
				//img.convertTo(img,CV_8UC1);
				img.copyTo(retorno(Rect(j*c,i*f,img.cols,img.rows)));
			}
		}
		return retorno;
	}
	//ENSUCIA UNA MATRIZ CON RUIDO IMPULSIVO
	//0: PIMIENTA, 1:SAL, 2: SAL,PIMIENTA
	//LAS IMAGENES DEL VECTOR REPRESENTAN CADA CANAL DE LA IMAGEN
	//SE LE  EL MISMO RUIDO A TODOS LOS CANALES
	//DEBEN SER 8UC
	vector<Mat> ruidoImpulsivo(vector<Mat> img, int modo, int probPintar){
		int f = img[0].rows, c = img[0].cols;
		//Mat ruido = Mat::zeros(f,c,CV_8UC1);
		vector<Mat> retorno(img.size());
		for (int z=0;z<img.size();z++) {//Copio las imagenes de entrada para no alterarlas
			retorno[z] = img[z].clone();
		}
		//Armo ruido
		for (int i=0;i<f;i++){
			for (int j=0;j<c;j++){
				int color = rand()%2, prob = (rand()%100)+1;
				if (prob<=probPintar){//Pinto
					for (int z=0;z<img.size();z++) {//Pinto cada canal, el mismo pixel con el mismo color
						if (modo == 2)//SAL - PIMIENTA
							retorno[z].at<uchar>(i,j) = color*255;
						if (modo == 0)//PIMIENTA
							retorno[z].at<uchar>(i,j) = 0;
						if (modo != 0 && modo != 2)//SAL
							retorno[z].at<uchar>(i,j) = 255;
					}
					
				}
			}
		}
		return retorno;
	}
	//FUNCION AUXILIAR PARA EL RUIDO GAUSSIANO
	//SUMA LAS MATRICES PERO NO PROMEDIA
	Mat ruidoGaussianoAux(Mat img1,Mat img2,int media){
		int f = img1.rows, c = img1.cols;
		Mat retorno = Mat::zeros(f,c,CV_8UC1);
		if(f==img2.rows && c==img2.cols){
			for (int i=0;i<f;i++){
				for (int j=0;j<c;j++){
					int sumando = img2.at<uchar>(i,j)-media;
					int valor = img1.at<uchar>(i,j) + sumando;
					if (valor>255)
						valor=255;
					if(valor<0){
						valor=0;
					}
					retorno.at<uchar>(i,j) = valor;
				}
			}
		}
		return retorno;
	}
	//ENSUCIA UN MATRIZ CON RUIDO GAUSSIANO
	vector<Mat> ruidoGaussiano(vector<Mat> img,int desvio,int media){
		int f=img[0].rows,c=img[0].cols;
		Mat ruido = Mat::zeros(f,c,CV_8UC1);
		randn(ruido,media,desvio);
		vector<Mat> retorno(img.size());
		for (int z=0;z<retorno.size();z++){
			retorno[z] = ruidoGaussianoAux(img[z].clone(),ruido,media);
		}
		return retorno;
	}
	
	float mediaGeometricaAux(Mat img){
		float e = (float) 1/(img.rows*img.cols), retorno=1;
		
		for (int i=0;i<img.rows;i++){
			for(int j=0;j<img.cols;j++){
				retorno *= img.at<float>(i,j);
			}
		}
		retorno = pow(retorno,e);
		return retorno;
	}
	
	//FILTRO MEDIA GEOMETRICA 
	//FILTRA UNA SOLA IMAGEN (1 SOLO CANAL)
	Mat mediaGeometrica(Mat img1, int n, int m){
		Mat img = img1.clone();
		//PASO A FLOAT SI ES UNSIGNED CHAR
		switch(img.depth()){
			case CV_8U: img.convertTo(img,CV_32F,1./125);
			normalize(img,img,0,1,CV_MINMAX);
			break;
		}
		
		int f=img.rows,c=img.cols;
		int a = m/2, b = n/2;	
		
		Mat aux = Mat::zeros(f+2*a,c+2*b,CV_32F);
		Mat filtrada = Mat::zeros(f,c,CV_32F);
		img.copyTo(aux(Rect(a,b,img.cols,img.rows)));
		
		for (int i=0;i<aux.rows-n;i++){
			for (int j=0;j<aux.cols-m;j++){
				filtrada.at<float>(i,j) = mediaGeometricaAux(aux(Rect(j,i,m,n)));
			}
		}

		return filtrada;
	}
	
	
	float mediaContraArmonicaAux(Mat img,int q){//IMAGEN DE FLOAT
		int f = img.rows, c = img.cols;
		float den=0 , num=0, retorno;
		for (int i=0;i<f;i++){
			for (int j=0;j<c;j++){
				float intensidad = img.at<float>(i,j);
				den+=pow(intensidad,q);
				num+=pow(intensidad,q+1);
			}
		}
		if (den==0){
			retorno = 0;
			cout<<"Denominador cero\n";
		}
		else
			retorno = num/den;
		return retorno;
	}
	
	//FILTRO MEDIA CONTRA ARMONICA
	//FILTRA UNA SOLA IMAGEN (1 SOLO CANAL)
	Mat mediaContraArmonica(Mat img1, int m, int n,int q){
		Mat img = img1.clone();
		//PASO A FLOAT SI ES UNSIGNED CHAR
		switch(img.depth()){
			case CV_8U: img.convertTo(img,CV_32F,1./125);
			normalize(img,img,0,1,CV_MINMAX);
			break;
		}
		
		int f=img.rows, c=img.cols;
		int a = m/2, b = n/2;
		
		Mat aux = Mat::zeros(f+2*a,c+2*b,CV_32F);//SE USA PARA EL PADDING
	
		Mat retorno = Mat::zeros(f,c,CV_32F);
		img.copyTo(aux(Rect(a,b,retorno.cols,retorno.rows)));
		
		//Borrar	
		for (int i=0;i<aux.rows-n;i++){
			for (int j=0;j<aux.cols-m;j++){
				float nada = mediaContraArmonicaAux(aux(Rect(j,i,m,n)),q);
				retorno.at<float>(i,j) = nada;
			}
		}
		return retorno;
	}
	
	//FILTRO MEDIANA
	//FILTRA UNA SOLA IMAGEN (1 SOLO CANAL)
	
	Mat burbuja(Mat v1,bool orden){//VECTOR FILA FLOTANTE
		Mat retorno,v=v1.clone();
		int n=v.cols;
		if (n==1){
			retorno = v;
		} else{
			float aux;
			for (int j=0;j<v.cols-1;j++){
				float valAct = v.at<float>(0,j),
					  valSig = v.at<float>(0,j+1);
				if (valAct<valSig) {//MINIMO A LO ULTIMO
					v.at<float>(0,j)=valSig;
					v.at<float>(0,j+1)=valAct;
				}
			}
			Mat recur = burbuja(v(Rect(0,0,v.cols-1,1)),orden);
			Mat auxMat = Mat::zeros(recur.rows,recur.cols+1,CV_32F);
			if (orden){//DESCENDENTE -> MINIMO A LO ULTIMO
				auxMat.at<float>(0,auxMat.cols-1) = v.at<float>(0,v.cols-1);
				recur.copyTo(auxMat(Rect(0,0,recur.cols,1)));
				retorno=auxMat;
			}else{//ASCENDENTE -> MINIMO AL COMIENZO
				auxMat.at<float>(0,0) = v.at<float>(0,v.cols-1);
				recur.copyTo(auxMat(Rect(1,0,recur.cols,1)));
				retorno=auxMat;
			}
		}
		return retorno;
	}
	
	
	//CONVIERTE UNA IMAGEN CUADRADA EN LINEAL (FILA)
	Mat cuadradaAlineal(Mat img){
		int f=img.rows, c=img.cols, l=f*c;
		Mat retorno = Mat::zeros(1,l,CV_32F);
		
		for (int i=0;i<img.rows;i++){
			for (int j=0;j<img.cols;j++){
				float valor = img.at<float>(i,j);
				retorno.at<float>(0,i*c+j) = valor;
			}
		}
		return retorno;
	}
	
	
	float medianaAux(Mat img){
		Mat imgl = cuadradaAlineal(img);
		imgl = burbuja(imgl,true);
		int posicion = ceil((float) imgl.cols/2);
		return imgl.at<float>(0,posicion);
	}
	
	Mat mediana(Mat img1, int m, int n){
		Mat img = img1.clone();
		//PASO A FLOAT SI ES UNSIGNED CHAR
		switch(img.depth()){
			case CV_8U: img.convertTo(img,CV_32F,1./125);
			normalize(img,img,0,1,CV_MINMAX);
			break;
		}
		
		int f=img.rows, c=img.cols;
		int a = m/2, b = n/2;
		
		Mat aux = Mat::zeros(f+2*a,c+2*b,CV_32F);//SE USA PARA EL PADDING
		
		Mat retorno = Mat::zeros(f,c,CV_32F);
		img.copyTo(aux(Rect(a,b,img.cols,img.rows)));
		
		//Borrar	
		for (int i=0;i<aux.rows-n;i++){
			for (int j=0;j<aux.cols-m;j++){
				float nada = medianaAux(aux(Rect(j,i,m,n)));
				retorno.at<float>(i,j) = nada;
			}
		}
		return retorno;
	}
	
	Mat puntoMedio(Mat img1, int m, int n){
		Mat img = img1.clone();
		//PASO A FLOAT SI ES UNSIGNED CHAR
		switch(img.depth()){
		case CV_8U: img.convertTo(img,CV_32F,1./125);
		normalize(img,img,0,1,CV_MINMAX);
		break;
		}
		
		int f=img.rows, c=img.cols;
		int a = m/2, b = n/2;
		
		Mat aux = Mat::zeros(f+2*a,c+2*b,CV_32F);//SE USA PARA EL PADDING
		
		Mat retorno = Mat::zeros(f,c,CV_32F);
		img.copyTo(aux(Rect(a,b,img.cols,img.rows)));
		
		//Borrar	
		for (int i=0;i<aux.rows-n;i++){
			for (int j=0;j<aux.cols-m;j++){
				Mat auxMat = cuadradaAlineal(aux(Rect(j,i,m,n)));
				auxMat = burbuja(auxMat,true);//vector fila ordenado
				float nada = (auxMat.at<float>(0,0) + auxMat.at<float>(0,auxMat.cols-1))/2;
				retorno.at<float>(i,j) = nada;
			}
		}
		return retorno;
	}
	
	Mat alfaRecortado(Mat img1, int m, int n, int d){
		Mat img = img1.clone();
		//PASO A FLOAT SI ES UNSIGNED CHAR
		switch(img.depth()){
			case CV_8U: img.convertTo(img,CV_32F,1./125);
			normalize(img,img,0,1,CV_MINMAX);
			break;
		}
		
		int f=img.rows, c=img.cols;
		int a = m/2, b = n/2;
		
		Mat aux = Mat::zeros(f+2*a,c+2*b,CV_32F);//SE USA PARA EL PADDING
		
		Mat retorno = Mat::zeros(f,c,CV_32F);
		img.copyTo(aux(Rect(a,b,img.cols,img.rows)));
		
		//Borrar	
		for (int i=0;i<aux.rows-n;i++){
			for (int j=0;j<aux.cols-m;j++){
				Mat auxMat = cuadradaAlineal(aux(Rect(j,i,m,n)));
				auxMat = burbuja(auxMat,true);//vector fila ordenado
				float nada = 0;
				
				for (int w=d-1;w<auxMat.cols-d;w++){
					nada += auxMat.at<float>(0,w);
				}
				nada=(float) nada/(n*m-2*d);
				retorno.at<float>(i,j) = nada;
			}
		}
		return retorno;
	}
	
	Mat sumaMatricesPromedio(vector<Mat> imagenes){
		int nImagenes = imagenes.size();
		Mat suma = Mat::zeros(imagenes[0].rows,imagenes[0].cols,CV_8UC1);
		for (int z=0;z<nImagenes;z++){
			Mat imagen = imagenes[z].clone();
			for (int i=0;i<imagen.rows;i++){
				for (int j=0;j<imagen.cols;j++){
					suma.at<uchar>(i,j) += imagen.at<uchar>(i,j)/nImagenes;
				}
			}
		}
		return suma;
	}
	
	//Brillo  medio
	double brilloMedio(Mat img){
		Mat aux = img.clone();
		switch(aux.depth()){
			case CV_32F: aux.convertTo(aux,CV_8UC1,255); 
			break;
		}
		
		double cont = 0;
		for(int i=0;i<aux.rows;i++){
			for (int j=0;j<aux.cols;j++){
				cont += (double) aux.at<uchar>(i,j);
			}
		}
		cont/=(aux.rows*aux.cols);		

		return cont;
	}
	
	//CENTRO DE MASA DE UN CONTORNO
	vector<Point> centroMasa(vector<vector<Point> > contornos){
		vector<Point> centros(contornos.size());
		for (int k=0;k<contornos.size();k++){
			Point cont(0,0);
			vector<Point> contorno = contornos[k];		
			for (int i=0;i<contorno.size();i++){
				Point punto = contorno[i];
				cont.x+=punto.x;
				cont.y+=punto.y;
			}
			cont.x/=contorno.size();
			cont.y/=contorno.size();
			
			centros[k] = cont;
		}
		return centros;
	}
	//POR LAS DUDAS
	//Dibujar contornos: drawContours(img, contornosR, k, Scalar(0,255,255),-1,8);
	//Calcular areas: double area = contourArea(contornosV[k]);
	
	//MouseCallback
//	void funcion(int event, int x, int y, int, void* userData){
//		if( event != EVENT_LBUTTONDOWN )
//			return;
//		Mat img = *( static_cast<Mat*>(userData)); //puede ser lo que sea, adentro le digo q es un matriz
//		//	info(img);
//		Vec3b c = img.at<Vec3b>(y,x);
//		int h = c[0], s = c[1], v = c[2];
//		
//		cout<<h<<" "<<s<<" "<<v<<"\n";
	
	//Para llamarlo: setMouseCallback("HSI",funcion,&colorHsi);
	
//	//CONTORNOS
//	vector<Vec4i> nosequees;
//	vector<vector<Point> > contornosV;
//	vector<vector<Point> > contornosR;
//	
//	findContours(verdes.clone(),contornosV,nosequees, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));		
//	findContours(rojas.clone(),contornosR,nosequees, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));		
	
	
	//----------------------FUNCIONES MIAS-------------------------------------
	
}

#endif /* PDI_FUNCTIONS_H */
