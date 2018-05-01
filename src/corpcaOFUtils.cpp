#include "corpcaOFUtils.h"
//#include "Utilities.cuh"
#include <opencv2/core/utility.hpp>
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

#ifdef CUDA
#include <cusolverDn.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_paraMeters.h"
#endif

using namespace cv;

#ifdef CUDA
using namespace cv::cuda;
#endif

namespace corpcaOFUtils

{
	cv::Mat scaledData(cv::Mat& dataIn, double minval, double maxval)
	{
		double mindata, maxdata;
		cv::minMaxLoc(dataIn, &mindata, &maxdata);
		cv::Mat dataOut = dataIn - mindata;
		cv::minMaxLoc(dataOut, &mindata, &maxdata);
		dataOut = (dataOut / (maxdata - mindata)) * (maxval - minval);
		dataOut += minval;
		return dataOut;
	}

	void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
	{
		std::vector<int> t_x, t_y;
		for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
		for (int i = ygv.start; i <= ygv.end; i++) t_y.push_back(i);

		cv::repeat(cv::Mat(t_x).reshape(1, 1), cv::Mat(t_y).total(), 1, X);
		cv::repeat(cv::Mat(t_y).reshape(1, 1).t(), 1, cv::Mat(t_x).total(), Y);

		X.convertTo(X, CV_32F);
		Y.convertTo(Y, CV_32F);
	}

	bool isFlowCorrect(Point2f u)
	{
		return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
	}

	cv::Vec3b computeColor(float fx, float fy)
	{
		static bool first = true;

		// relative lengths of color transitions:
		// these are chosen based on perceptual similarity
		// (e.g. one can distinguish more shades between red and yellow
		//  than between yellow and green)
		const int RY = 15;
		const int YG = 6;
		const int GC = 4;
		const int CB = 11;
		const int BM = 13;
		const int MR = 6;
		const int NCOLS = RY + YG + GC + CB + BM + MR;
		static Vec3i colorWheel[NCOLS];

		if (first)
		{
			int k = 0;

			for (int i = 0; i < RY; ++i, ++k)
				colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

			for (int i = 0; i < YG; ++i, ++k)
				colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

			for (int i = 0; i < GC; ++i, ++k)
				colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

			for (int i = 0; i < CB; ++i, ++k)
				colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

			for (int i = 0; i < BM; ++i, ++k)
				colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

			for (int i = 0; i < MR; ++i, ++k)
				colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

			first = false;
		}

		const float rad = sqrt(fx * fx + fy * fy);
		const float a = atan2(-fy, -fx) / (float)CV_PI;

		const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
		const int k0 = static_cast<int>(fk);
		const int k1 = (k0 + 1) % NCOLS;
		const float f = fk - k0;

		Vec3b pix;

		for (int b = 0; b < 3; b++)
		{
			const float col0 = colorWheel[k0][b] / 255.0f;
			const float col1 = colorWheel[k1][b] / 255.0f;

			float col = (1 - f) * col0 + f * col1;

			if (rad <= 1)
				col = 1 - rad * (1 - col); // increase saturation with radius
			else
				col *= .75; // out of range

			pix[2 - b] = static_cast<uchar>(255.0 * col);
		}

		return pix;
	}

	void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion)
	{
		dst.create(flowx.size(), CV_8UC3);
		dst.setTo(Scalar::all(0));

		// determine motion range:
		float maxrad = maxmotion;

		if (maxmotion <= 0)
		{
			maxrad = 1;
			for (int y = 0; y < flowx.rows; ++y)
			{
				for (int x = 0; x < flowx.cols; ++x)
				{
					Point2f u(flowx(y, x), flowy(y, x));

					if (!isFlowCorrect(u))
						continue;

					maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
				}
			}
		}

		for (int y = 0; y < flowx.rows; ++y)
		{
			for (int x = 0; x < flowx.cols; ++x)
			{
				Point2f u(flowx(y, x), flowy(y, x));

				if (isFlowCorrect(u))
					dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
			}
		}
	}

#ifdef CUDA
	static void showFlow(const char* name, const GpuMat& d_flow)
	{
		GpuMat planes[2];
		cuda::split(d_flow, planes);

		Mat flowx(planes[0]);
		Mat flowy(planes[1]);

		Mat out;
		drawOpticalFlow(flowx, flowy, out, 10);

		imshow(name, out);
	}

	cv::Mat writeFlow(const GpuMat& d_flow)
	{
		GpuMat planes[2];
		cuda::split(d_flow, planes);

		Mat flowx(planes[0]);
		Mat flowy(planes[1]);

		Mat out;
		drawOpticalFlow(flowx, flowy, out, 10);
		return out;
	}
#else
	static void showFlow(const char* name, const cv::Mat& d_flow)
	{
		Mat planes[2];
		cv::split(d_flow, planes);
		cv::Mat flowx(planes[0]); Mat flowy(planes[1]);
		Mat out;
		drawOpticalFlow(flowx, flowy, out, 10);

		imshow(name, out);
	}

	cv::Mat writeFlow(const Mat& d_flow)
	{
		Mat planes[2];
		cv::split(d_flow, planes);

		Mat flowx(planes[0]);
		Mat flowy(planes[1]);

		Mat out;
		drawOpticalFlow(flowx, flowy, out, 10);
		return out;
	}
#endif
	cv::Mat linearMotionCompensation(const cv::Mat& img, const cv::Mat& hor_ind, const cv::Mat& ver_ind)
	{
		int numRows = img.rows;
		int numCols = img.cols;

		cv::Mat vRow = cv::Mat::zeros(ver_ind.rows * ver_ind.cols, 1, CV_32F);
		cv::Mat hRow = cv::Mat::zeros(hor_ind.rows * hor_ind.cols, 1, CV_32F);
		for (int i = 0; i < ver_ind.cols; i++)
		{
			int rowIndex = i * numRows;
			ver_ind.col(i).copyTo(vRow.rowRange(rowIndex, rowIndex + numRows));
			hor_ind.col(i).copyTo(hRow.rowRange(rowIndex, rowIndex + numRows));
		}

		cv::Mat vert_floor = cv::Mat::zeros(vRow.rows, vRow.cols, CV_32F);
		cv::Mat hori_floor = cv::Mat::zeros(hRow.rows, hRow.cols, CV_32F);

		if (vert_floor.size() == hori_floor.size())
		{
			for (int i = 0; i < vert_floor.rows; i++)
			{
				vert_floor.at<float>(i, 0) = cv::min((double)cvFloor(vRow.at<float>(i, 0)), (double)numRows - 2);
				hori_floor.at<float>(i, 0) = cv::min((double)cvFloor(hRow.at<float>(i, 0)), (double)numCols - 2);
			}
		}

		cv::Mat vert_ceil = vert_floor + 1;
		cv::Mat hori_ceil = hori_floor + 1;

		cv::Mat topLeftCorner = vert_floor + (numRows) * (hori_floor);
		cv::Mat bottomLeftCorner = vert_ceil + (numRows) * (hori_floor);
		cv::Mat bottomRightCorner = vert_ceil + (numRows) * (hori_ceil);
		cv::Mat topRightCorner = vert_floor + (numRows) * (hori_ceil);

		//column interpolation
		cv::Mat ITopLeft = cv::Mat::zeros(numRows * numCols, 1, CV_32F);
		cv::Mat ITopRight = cv::Mat::zeros(numRows * numCols, 1, CV_32F);
		cv::Mat IBottomLeft = cv::Mat::zeros(numRows * numCols, 1, CV_32F);
		cv::Mat IBottomRight = cv::Mat::zeros(numRows * numCols, 1, CV_32F);


		cv::Mat linImg = cv::Mat::zeros(numRows * numCols, 1, CV_32F);
		for (int i = 0; i < numCols; i++)
		{
			int rowIndex = i * numRows;
			img.col(i).copyTo(linImg.rowRange(rowIndex, rowIndex + numRows));
		}

		for (int i = 0; i < linImg.rows; i++)
		{
			ITopLeft.at<float>(i, 0) = linImg.at<float>(topLeftCorner.at<float>(i, 0), 0);
			ITopRight.at<float>(i, 0) = linImg.at<float>(topRightCorner.at<float>(i, 0), 0);
			IBottomLeft.at<float>(i, 0) = linImg.at<float>(bottomLeftCorner.at<float>(i, 0), 0);
			IBottomRight.at<float>(i, 0) = linImg.at<float>(bottomRightCorner.at<float>(i, 0), 0);
		}

		cv::Mat I1_left = ITopLeft.mul(vert_ceil - vRow) + IBottomLeft.mul(vRow - vert_floor);
		cv::Mat I1_right = ITopRight.mul(vert_ceil - vRow) + IBottomRight.mul(vRow - vert_floor);

		cv::Mat I1_temp = I1_left.mul(hori_ceil - hRow) + I1_right.mul(hRow - hori_floor);

		//cv::Mat I = I1_temp.reshape(0, numRows);
		cv::Mat I = cv::Mat::zeros(numRows, numCols, CV_32F);

		for (int i = 0; i < numCols; i++)
		{
			int rowIndex = i * numRows;
			I1_temp.rowRange(rowIndex, rowIndex + numRows).copyTo(I.col(i));
		}

		I.convertTo(I, CV_64F);
		return I;
	}

	double lnorm(const cv::Mat& matrix, double l)
	{
		double result = 0.0;
		for (unsigned int i = 0; i < matrix.rows; ++i)
		{
			for (unsigned int j = 0; j < matrix.cols; ++j)
			{
				double value = matrix.at<double>(i, j);
				result += std::pow(value, l);
			}
		}
		return std::pow(result, (1 / l));
		cv::Mat transMat;
		cv::transpose(matrix, transMat);
		result = std::sqrt(cv::sum(cv::Mat::diag(transMat * matrix))[0]);

		return result;
	}

	double frobNorm(cv::Mat& M)
	{
		cv::Mat M_t;
		cv::transpose(M, M_t);
		double sum = cv::trace(M_t * M)[0];

		double norm = std::sqrt(sum);
		return norm;
	}

	cv::Mat signum(cv::Mat src)
	{
		cv::Mat z = Mat::zeros(src.size(), src.type());
		cv::Mat a = (z < src) & 1;
		cv::Mat b = (src < z) & 1;

		Mat dst;
		cv::addWeighted(a, 1.0, b, -1.0, 0.0, dst, CV_64F);
		return dst;
	}

	bool readBinary(const std::string& filename, cv::Mat& output)
	{
		std::fstream ifs(filename, std::ios::in | ::ios::binary);
		int length;
		if (ifs)
		{
			ifs.seekg(0, ifs.end);
			length = ifs.tellg();
			ifs.seekg(0, ifs.beg);
		}

		char* buffer;
		buffer = new char[length];

		ifs.read(buffer, length);

		ifs.close();

		double* double_values = (double*)buffer;

		std::vector<double> buffer2(double_values, double_values + (length / sizeof(double)));
		cv::Mat temp = cv::Mat::zeros(output.rows, output.cols, CV_64F);

		int i, j, t = 0;
		for (j = 0; j < output.cols; j++)
		{
			for (i = 0; i < output.rows; i++)
			{
				temp.at<double>(i, j) = buffer2.at(i + j * output.rows);
			}
		}
		temp.copyTo(output);
		return true;
	}
}