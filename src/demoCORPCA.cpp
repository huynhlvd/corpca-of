#include "corpca.h"

using namespace cv;

int main()
{
	cv::Mat A = cv::Mat::ones(100, 100, CV_32F);
	cv::Mat sigA = signum(A);
	
	return 0;
}