// system includes
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

// library includes
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#define EXPORT_FUNC  __declspec(dllexport)

using namespace std;
using namespace cv;

EXPORT_FUNC bool readBinary(const std::string& filename, cv::Mat& output);
EXPORT_FUNC double frobNorm(cv::Mat& M);
EXPORT_FUNC void incSVD(const cv::Mat& v, const cv::Mat& U0, const cv::Mat& S0, const cv::Mat& V0, cv::Mat& U1, cv::Mat& S1, cv::Mat& V1);
EXPORT_FUNC void corpca(const cv::Mat& yt, const cv::Mat& Phi, const cv::Mat& invPhi, const cv::Mat& Ztm1, cv::Mat& Btm1, cv::Mat& xt, cv::Mat& vt, cv::Mat& Zt, cv::Mat& Bt);
EXPORT_FUNC cv::Mat softMSI(const Mat& x, double lambda, double L, const Mat& Wk, const Mat& Z);
EXPORT_FUNC cv::Mat signum(cv::Mat src);
EXPORT_FUNC cv::Mat proxMat(const Mat& x, const Mat& A, const Mat& W, double lambda, double L);


