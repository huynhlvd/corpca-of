/*
2017
Author: P.V.Srivatsa, srivatsa.pv@live.com
*/



#ifndef RAMSIA_H
#define RAMSIA_H

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"

//RAMSIA algorithm
cv::Mat ramsi(const cv::Mat& A, const cv::Mat& b, const cv::Mat& Zin);
//Compute sum of norm1
double sum_norm1(const cv::Mat& Wk, const cv::Mat& xk, const cv::Mat& Z);
//Proximal operator
cv::Mat proxMat(const cv::Mat& x, const cv::Mat& A, const cv::Mat& W, double lambda, double L);
//Soft thresholding operator
cv::Mat softMSI(const cv::Mat& x, double lambda, double L, const cv::Mat& Wk, const cv::Mat& Z);

#endif //RAMSIA_H