/*
2017
Author: P.V.Srivatsa, srivatsa.pv@live.com
*/


#ifndef CORPCAOF_UTILS_H
#define CORPCAOF_UTILS_H

//system includes
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <chrono>

//opencv includes
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"

#define CUDA
#undef CUDA

#ifdef SHARE_LIB
#define	EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __declspec(dllimport)
#endif

using namespace std;
using namespace cv;
#ifdef CUDA
using namespace cv::cuda;
#endif

using ms = chrono::milliseconds;
using get_time = chrono::steady_clock;

namespace corpcaOFUtils
{
	//to scale a matrix to a range specified by minval and maxval
	cv::Mat scaledData(cv::Mat& dataIn, double minval, double maxval);

	//C++ equivalent of matlab meshgrid function
	void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y);

	//validate optical flow point
	inline bool isFlowCorrect(Point2f u);

	//compute color for a given optical flow vector
	static Vec3b computeColor(float fx, float fy);

	//visualize optical flow
	static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1);

#ifdef CUDA
	//display optical flow
	static void showFlow(const char* name, const GpuMat& d_flow);

	//write optical flow to file
	cv::Mat writeFlow(const GpuMat& d_flow);
#else
	//display optical flow
	static void showFlow(const char* name, const Mat& d_flow);

	//write optical flow to file
	cv::Mat writeFlow(const Mat& d_flow);
#endif
	//linear motion compensation on image using horizontal and vertical indices
	cv::Mat linearMotionCompensation(const cv::Mat& img, const cv::Mat& hor_ind, const cv::Mat& ver_ind);

	//estimate l-norm for a given matrix
	double lnorm(const cv::Mat& matrix, double l = 2);

	//estimate Frobenius norm for a given matrix
	double frobNorm(cv::Mat& M);

	//signum function to return sign if different and 0 if same
	cv::Mat signum(cv::Mat src);

	//read Binary files into cv::Mat
	bool readBinary(const std::string& filename, cv::Mat& output);
}

#endif //CORPCAOF_UTILS_H