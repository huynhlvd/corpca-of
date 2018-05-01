/*
2017
Author: P.V.Srivatsa, srivatsa.pv@live.com
*/


#ifndef PRIOR_INIT_H
#define PRIOR_INIT_H

// library includes
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"

//Initilization of prior by specifying videopath
void priorInitialization(cv::String videopath, int batch_size, int MAXITER, double RESIZE, cv::Mat& B0, cv::Mat& Z0, int iter);
//Initilization of prior by specifying a list of video files
void priorInitialization(std::vector<cv::String> videoFiles, int batch_size, int MAXITER, double RESIZE, cv::Mat& B0, cv::Mat& Z0, int iter);


#endif //PRIOR_INIT_H
