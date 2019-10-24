#include<iostream>
#include "myCImg.h"

int main (int argc, char *argv[]) {
	char filename[32];
	for( int i=1; i<3; i++ ) {
		sprintf(filename, "%i.jpg", i);
		CImg<double> original( filename );
		CImg<double> filtrada = AlfaRecortado( original, 5, 5, 2 );
		CImg<bool> segmentada = InundarInverso( filtrada.get_blur(5), 0.35 );
		CImg<double> hough = TransformadaHough( segmentada );
		CImg<double> linea = InversaHough( hough, 100 );
		CImgList<double> lista( /*original, filtrada,*/ segmentada, /*hough,*/ linea.mul(segmentada) + filtrada );
		lista.display("Pista", false);
	}
	
	return 0;
}

