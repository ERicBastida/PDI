#include<iostream>
#include "pdi_functions.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <random>
using namespace cv;
//Smooth Filters
cv::Mat mean_filter(cv::Mat &im,int msize){//msize must be odd
	cv::Mat result;
	blur(im, result, Size(msize,msize));
	return result;
}
cv::Mat geometric_filter(cv::Mat &im,int msize){//msize must be odd
	double acum;
	cv::Mat result=im.clone(),orig;
	copyMakeBorder(result,result,(msize-1)/2, (msize-1)/2, (msize-1)/2, (msize-1)/2, BORDER_REPLICATE);
	orig=result.clone();
	int mside=(msize-1)/2;
	for(int i=mside;i<result.rows-mside;i++)
		for(int j=mside;j<result.cols-mside;j++){
		acum=1;
		for(int ii=i-mside;ii<=i+mside;ii++){
			for(int jj=j-mside;jj<=j+mside;jj++){
				acum*=orig.at<uchar>(ii,jj);
			}
		}
		acum=pow(acum,1./(msize*msize));
		result.at<uchar>(i,j)=acum;
	}
	result=result(Rect(mside,mside,im.cols,im.rows));
	return result;
	
}

cv::Mat contraHarmonic_filter(cv::Mat &im,int msize, float Q){//msize must be odd
	float acum1,acum2;
	cv::Mat result=im.clone(),orig;
	copyMakeBorder(result,result,(msize-1)/2, (msize-1)/2, (msize-1)/2, (msize-1)/2, BORDER_REPLICATE);
	orig=result.clone();
	int mside=(msize-1)/2;
	for(int i=mside;i<result.rows-mside;i++)
		for(int j=mside;j<result.cols-mside;j++){
		acum1=0;acum2=0;
		for(int ii=i-mside;ii<=i+mside;ii++){
			for(int jj=j-mside;jj<=j+mside;jj++){
				acum1+=pow(orig.at<uchar>(ii,jj),Q+1);
				acum2+=pow(orig.at<uchar>(ii,jj),Q);
			}
		}
		result.at<uchar>(i,j)=acum1/acum2;
	}
	result=result(Rect(mside,mside,im.cols,im.rows));
	return result;
	
}

cv::Mat median_filter(cv::Mat &im,int msize){//msize must be odd
	cv::Mat result;
	medianBlur(im,result,msize);
	return result;
}

cv::Mat midpoint_filter(cv::Mat &im,int msize){//msize must be odd
	uchar v[msize*msize];
	cv::Mat result=im.clone(),orig;
	copyMakeBorder(result,result,(msize-1)/2, (msize-1)/2, (msize-1)/2, (msize-1)/2, BORDER_REPLICATE);
	orig=result.clone();
	int k,t=(msize*msize)-1,mside=(msize-1)/2;
	for(int i=mside;i<result.rows-mside;i++)
		for(int j=mside;j<result.cols-mside;j++){
		k=0;
		for(int ii=i-mside;ii<=i+mside;ii++){
			for(int jj=j-mside;jj<=j+mside;jj++){
				v[k++]=orig.at<uchar>(ii,jj);
			}
		}
		std::sort(v, v+(t+1));
		result.at<uchar>(i,j)=0.5*(v[0]+v[t]);
	}
	result=result(Rect(mside,mside,im.cols,im.rows));
	return result;
	
}

cv::Mat alphatrimmed_filter(cv::Mat &im,int msize, int d){//msize must be odd
	uchar v[msize*msize];
	cv::Mat result=im.clone(),orig;
	copyMakeBorder(result,result,(msize-1)/2, (msize-1)/2, (msize-1)/2, (msize-1)/2, BORDER_REPLICATE);
	orig=result.clone();
	float t=1./((msize*msize)-d);
	int a,k,t2=(msize*msize)-1,mside=(msize-1)/2;
	for(int i=mside;i<result.rows-mside;i++)
		for(int j=mside;j<result.cols-mside;j++){
		k=0;
		for(int ii=i-mside;ii<=i+mside;ii++){
			for(int jj=j-mside;jj<=j+mside;jj++){
				v[k++]=orig.at<uchar>(ii,jj);
			}
		}
		std::sort(v, v+(msize*msize));
		a=0;
		for(int i=d/2;i<=t2-(d/2);i++)
			a=a+v[i];
		result.at<uchar>(i,j)=floor(t*a);
	}
	result=result(Rect(mside,mside,im.cols,im.rows));
	return result;
	
}
//Sharp Filters
// sharpen image using "unsharp mask" algorithm
cv::Mat high_boost_filter(cv::Mat &im,double threshold, double sigma, double amount){
	Mat blurred;
	GaussianBlur(im, blurred, Size(), sigma, sigma);
	Mat lowContrastMask = abs(im - blurred) < threshold;
	Mat sharpened = im*(1+amount) + blurred*(-amount);
	im.copyTo(sharpened, lowContrastMask);
	return sharpened;
}
cv::Mat sharp_filter(cv::Mat &im,int kernel_size, double c,int kernelnumber){
	cv::Mat im_filtrada;
	//Kernels suma 0
	cv::Mat kernel = Mat::ones( kernel_size, kernel_size, CV_32F )*(-1) ;
	cv::Mat kernel2 = Mat::ones( kernel_size, kernel_size, CV_32F )*0;
	kernel.at<float>(1,1) = 8.f;
	kernel2.at<float>(0,1) = -1.f;kernel2.at<float>(1,0) = -1.f;kernel2.at<float>(1,2) = -1.f;
	kernel2.at<float>(2,1) = -1.f;kernel2.at<float>(1,1) = 4.f;
	//Kernels suma 1
	cv::Mat kernel3 = Mat::zeros( kernel_size, kernel_size, CV_32F );
	kernel3.at<float>(0,1)=-1.f;kernel3.at<float>(1,0)=-1.f;
	kernel3.at<float>(1,2)=-1.f;kernel3.at<float>(2,1)=-1.f;kernel3.at<float>(1,1)=4.f;
	cv::Mat kernel4=-kernel;
	//filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT );
	switch(kernelnumber){
	case 1:
		kernel=kernel;
		break;
	case 2:
		kernel=kernel2;
		break;
	case 3:
		kernel=kernel3;
		break;
	case 4:
		kernel=kernel4;
		c=-c;
		break;
	default:
		kernel=kernel;
	}
	filter2D(im, im_filtrada,-1,kernel);
	im_filtrada=im+c*im_filtrada;
	return im_filtrada;
}
