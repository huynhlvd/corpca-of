#include "priorInititialization.h"
#include "inexact_alm_rpca.h"
#include "corpcaOFUtils.h"
#include <iostream>

using namespace std;
using namespace cv;
using namespace corpcaOFUtils;

void priorInitialization(cv::String videopath, int batch_size, int MAXITER, double RESIZE, cv::Mat& B0, cv::Mat& Z0, int iter)
{
	vector<String> filenames;

	cv::glob(videopath, filenames);

	int frame_count = 0;

	std::random_shuffle(filenames.begin(), filenames.end());

	cv::Mat myImage;
	cv::Mat resImg;
	cv::Mat batch_buf;
	int dim;
	double RPCA_lambda;
	for (int t = 0; t < filenames.size(); t++)
	{
		frame_count++;
		myImage = imread(filenames[t], cv::IMREAD_GRAYSCALE);
		cv::resize(myImage, resImg, Size(), RESIZE, RESIZE, cv::INTER_CUBIC);
		resImg = resImg.reshape(0, resImg.rows * resImg.cols);
		int k = frame_count % batch_size;
		if (k == 0)
			k = batch_size;

		if (frame_count == 1)
		{
			dim = resImg.rows * resImg.cols;
			batch_buf = cv::Mat(dim, batch_size, CV_64F);
			RPCA_lambda = 1 / sqrt(dim);
		}

		resImg.copyTo(batch_buf.col(k - 1));

	}

	inexact_alm_rpca(batch_buf, RPCA_lambda, -1, MAXITER, B0, Z0, iter);
}

void priorInitialization(std::vector<cv::String> videoFiles, int batch_size, int MAXITER, double RESIZE, cv::Mat& B0, cv::Mat& Z0, int iter)
{
	vector<String> filenames = videoFiles;

	std::random_shuffle(filenames.begin(), filenames.end()/*begin() + 100*/);

	cv::Mat myImage;
	cv::Mat resImg;
	cv::Mat batch_buf;
	int dim = 0;
	double RPCA_lambda = 0;
	int frame_count = 0;
	for (int t = 0; t < filenames.size(); t++)
	{
		frame_count++;
		myImage = imread(filenames[t], cv::IMREAD_GRAYSCALE);
		cv::resize(myImage, resImg, Size(), RESIZE, RESIZE, cv::INTER_CUBIC);
		resImg = resImg.reshape(0, resImg.rows * resImg.cols);
		int k = frame_count % batch_size;
		if (k == 0)
			k = batch_size;

		if (frame_count == 1)
		{
			dim = resImg.rows * resImg.cols;
			batch_buf = cv::Mat(dim, batch_size, CV_64F);
			RPCA_lambda = 1 / sqrt(dim);
		}

		resImg.copyTo(batch_buf.col(k - 1));

	}

	inexact_alm_rpca(batch_buf, RPCA_lambda, -1, MAXITER, B0, Z0, iter);
}