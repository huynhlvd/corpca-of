/*
2017
Author: P.V.Srivatsa, srivatsa.pv@live.com
*/



#ifndef INEXACT_ALM_RPCA_H
#define INEXACT_ALM_RPCA_H

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"

//Inexact ALM RPCA
void inexact_alm_rpca(const cv::Mat &D, double lambda, double tol, int maxIter, cv::Mat& A_hat, cv::Mat& E_hat, int iter);

#endif //INEXACT_ALM_RPCA_H