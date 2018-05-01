/*
2017
Author: P.V.Srivatsa, srivatsa.pv@live.com
*/

#ifndef CORPCAOF_H
#define CORPCAOF_H

// system includes
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

// library includes
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/superres/optical_flow.hpp"

#include "VideoSourceManager.h"

#ifdef SHARE_LIB
#define	EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __declspec(dllimport)
#endif

//#define CUDA

//Incremental SVD
void incSVD(const cv::Mat& v, const cv::Mat& U0, const cv::Mat& S0, const cv::Mat& V0, cv::Mat& U1, cv::Mat& S1, cv::Mat& V1);
//CORPCA function
EXPORT_FUNC void corpca(const cv::Mat& yt, const cv::Mat& Phi, const cv::Mat& invPhi, cv::Mat& Ztm1, cv::Mat& Btm1, cv::Mat& xt, cv::Mat& vt, cv::Mat& Zt, cv::Mat& Bt);
//CORPCA-OF function
EXPORT_FUNC void corpcaOF(VideoSourceManager* vsManager, cv::Mat& fgCorpcaOF, cv::Mat& bgCorpcaOF);
//Convert a column/row matrix into image by specifying the number of rows
void convertMatToImage(cv::Mat& mat, cv::Mat& img, int numRows);
//Compute OF using LDOF (CUDA implementation)
void computeOF(cv::Mat& x1, cv::Mat& xm1, cv::Mat& flowx, cv::Mat& flowy, cv::Mat& outOF);
//Compute optical flow indices from optical flow vectors
void computeFlowIndices(cv::Mat& flowx, cv::Mat& flowy, int width, int height, float mulFactor, cv::Mat& indX, cv::Mat& indY);

#endif //CORPCAOF_H
