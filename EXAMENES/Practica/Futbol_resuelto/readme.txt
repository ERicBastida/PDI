//INCLUDES
void ColorMean(cv::Mat im, cv::Scalar &color);
void ConnectedComponents (cv::Mat &imagen );
void ContrastStretching(cv::Mat image, cv::Mat &new_image, int r1, int s1, int r2, int s2);
void CutterImage(cv::Mat im,cv::Mat &out);
void CutterMask(cv::Mat mask,cv::Mat &mask_cutted,cv::Mat &im);
cv::Mat mean_filter(cv::Mat &im,int msize);
cv::Mat geometric_filter(cv::Mat &im,int msize);
cv::Mat contraHarmonic_filter(cv::Mat &im,int msize, float Q);
cv::Mat median_filter(cv::Mat &im,int msize);
cv::Mat midpoint_filter(cv::Mat &im,int msize);
cv::Mat alphatrimmed_filter(cv::Mat &im,int msize, int d);
cv::Mat high_boost_filter(cv::Mat &im,double threshold, double sigma, double amount);
cv::Mat sharp_filter(cv::Mat &im,int kernel_size, double c,int kernelnumber);
void Huang(cv::Mat im,cv::Mat &cdst,cv::Mat &dst,double threshold,bool vertical=0,bool horizontal=0);
void HuangP(cv::Mat im);
void HuangP(cv::Mat im,cv::Mat &cdst,cv::Mat &dst,double threshold,double angle,double thresholdAngle);
void RegionGrowing(cv::Mat im, cv::Mat &result, cv::Mat &mask, int x, int y, double lodiff, double updiff, int neighbor=4);
void RoI(cv::Mat im,cv::Mat &roi,int x, int y, int w, int h);
void segmentator(cv::Mat src,cv::Mat &segment,cv::Mat &mask,int cx,int cy);
void segmentator(cv::Mat src2,cv::Mat &segment2,cv::Mat &mask2,cv::Mat &mask2_inv,int cx,int cy,int ancho2,int alto2);
void segmentator(cv::Mat src2,cv::Mat &segment2,cv::Mat &mask2,cv::Mat &mask2_inv,int h,int s,int v,int ancho2,int alto2);
void Skeleton(cv::Mat img,cv::Mat &skel);
void view_coordinates(cv::Mat im);
void ConvexHull(cv::Mat src);
void ConvexHull(cv::Mat src,int threshold_ch,cv::Mat &output);
//eroding
unsigned char e[6][6]={{0,0,0,0,0,1},
				{0,0,0,0,1,0},
				{0,0,0,1,0,0},
				{0,0,1,0,0,0},
				{0,1,0,0,0,0},
				{1,0,0,0,0,0}};
	cv::Mat ee(6,6,CV_8U,e);
	e=getStructuringElement(MORPH_RECT,Size(33,33));
	erode(im,result,ee);
//add border to image
copyMakeBorder(InputArray src, OutputArray dst, int top, int bottom, int left, int right, int borderType, const Scalar& value=Scalar() )
Parameters:	

    src – Source image.
    dst – Destination image of the same type as src and the size Size(src.cols+left+right, src.rows+top+bottom) .
    top –
    bottom –
    left –
    right – Parameter specifying how many pixels in each direction from the source image rectangle to extrapolate. For example, top=1, bottom=1, left=1, right=1 mean that 1 pixel-wide border needs to be built.
    borderType – Border type. See borderInterpolate() for details.
    value – Border value if borderType==BORDER_CONSTANT .


