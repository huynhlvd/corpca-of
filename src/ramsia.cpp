#include "ramsia.h"
#include "corpcaOFUtils.h"

using namespace corpcaOFUtils;
using namespace cv;

enum stoppingCriterion
{
	STOPPING_SPARSE_SUPPORT = 1,
	STOPPING_OBJECTIVE_VALUE,
	STOPPING_SUBGRADIENT
};

cv::Mat ramsi(const cv::Mat& A, const cv::Mat& b, const cv::Mat& Zin)
{
	//Initialization 
	int n = A.cols;
	unsigned int max_iterations = 5000;
	double tolerance = 1e-5;
	double t_k = 1;
	cv::Mat invA(A);
	invert(A, invA, cv::DECOMP_SVD);
	cv::Mat G(invA.rows, A.cols, CV_32FC4);
	G = invA * A;
	cv::Mat eigenG(G);
	eigen(G, eigenG);

	double min_val, max_val;
	cv::minMaxLoc(eigenG, &min_val, &max_val);
	double L0 = max_val;
	unsigned int numIter = 0;
	cv::Mat c(invA.rows, b.cols, CV_32FC4);
	c = invA * b;

	double lambda0 = 0.5 * L0 * cv::norm(c, cv::NORM_INF);
	double eta = 0.95;

	double lambda_bar = 1e-5;

	cv::Mat xk;//(A.rows, 1, CV_64F, double(0));
			   //cv::solve(A.t(), b, xk);
			   //xk = c; 
	xk = cv::Mat::zeros(A.cols, b.cols, CV_64F);
	double lambda = lambda0;
	double L = L0;


	//Side info

	int J = Zin.cols + 1;
	cv::Mat Z(n, J, CV_64F, double(0));

	//cv::Range colRange = cv::Range(0, min(srcMat.cols, dstMat.cols));	
	Z.col(0).setTo(0);
	Zin.copyTo(Z.colRange(1, Z.cols));

	//Initialization
	cv::Mat Wk(n, J, CV_64F, double(0));
	Wk.col(0).setTo(double(1));

	bool keep_going = true;
	cv::Mat beta(J, 1, CV_64F, double(0));
	beta.at<double>(0, 0) = 1.0;

	double f_norm = cv::norm(b - A * xk, cv::NORM_L2);
	double f = 0.5 * pow(f_norm, 2) + lambda_bar * sum_norm1(Wk, xk, Z); 

	cv::Mat xkm1(xk); //x(k-1)
	double epsilon = 0.5;
	double epsilonBeta = 1e-20;
	cv::Mat uk(xk);
	cv::Mat Wkp1(Wk);

	cv::Mat u_kp1(xk.rows, xk.cols, CV_64F, double(0));
	int stopCriteria = STOPPING_OBJECTIVE_VALUE;


	while (keep_going && (numIter < max_iterations))
	{
		numIter++;
		cv::Mat temp = G * uk - c; //gradient of f at uk
		cv::Mat gk = uk - (1 / L)*temp;
		xk = softMSI(gk, lambda, L, Wk, Z);

		switch (stopCriteria)
		{
		case STOPPING_OBJECTIVE_VALUE: //this is what is being used
									   /*stopping criteria is computed based on the relative variation of the objective function*/
			double prev_f = f;
			cv::Mat difff = b - A * xk;

			double fnorm = cv::norm(difff, cv::NORM_L2);
			double sumnorm = sum_norm1(Wk, xk, Z);
			f = 0.5 * pow(fnorm, 2) + lambda_bar * sumnorm;
			double criterionObjective = abs(f - prev_f) / (prev_f);
			keep_going = criterionObjective > tolerance;
			break;
		}

		lambda = max(eta * lambda, lambda_bar);

		//Weight updating
		for (int j = 0; j < J; j++)
		{
			cv::Mat denom = epsilon + abs(xk - Z.col(j));
			Wkp1.col(j) = 1 / (denom.col(0));
			Wkp1.col(j) = Wkp1.col(j) * (n / (cv::sum(Wkp1.col(j))[0]));
		}

		for (int j = 0; j < J; j++)
		{
			double denom_scalar = 0;
			for (int i = 0; i < J; i++)
			{
				denom_scalar += epsilonBeta + sum_norm1(Wkp1.col(j), xk, Z.col(j)) / (epsilonBeta + sum_norm1(Wkp1.col(i), xk, Z.col(i)));
				beta.at<double>(j, 0) = 1 / denom_scalar;
			}
		}

		//New weights
		for (int j = 0; j < J; j++)
		{
			Wkp1.col(j) = beta.at<double>(j, 0) * Wkp1.col(j);
		}

		//Update values for next iteration
		double t_kp1 = 0.5 * (1 + sqrt(1 + 4 * t_k * t_k));
		u_kp1 = xk + ((t_k - 1) / t_kp1)	* (xk - xkm1);

		//Next iteration
		xkm1 = xk;

		Wk = Wkp1;

		uk = u_kp1;
		t_k = t_kp1;


	}
	cv::Mat x_hat(xk);
	return x_hat;
}

double sum_norm1(const Mat& Wk, const Mat& xk, const Mat& Z)
{
	double sum = 0;
	int J = Z.cols;
	for (int j = 0; j < J; j++)
	{
		sum += cv::norm(Wk.col(j).mul(xk - Z.col(j)), cv::NORM_L1);
	}
	return sum;
}

cv::Mat proxMat(const Mat& x, const Mat& A, const Mat& W, double lambda, double L)
{
	int J = A.cols - 3;
	cv::Mat S(x.rows, A.cols - 1, CV_64F, double(0));
	cv::Mat P(x.rows, (A.cols - 1) * 2, CV_64F, double(0));

	for (int m = 0; m < S.cols; m++)
	{
		for (int j = 1; j < (J + 2); j++)
		{
			double boolfunc = pow(-1, (m - 1 < j - 1));
			S.col(m) = S.col(m) + W.col(j) * boolfunc;
		}
		S.col(m) = S.col(m) * (lambda / L);
		P.col(2 * m) = A.col(m) + S.col(m);
		P.col(2 * m + 1) = A.col(m + 1) + S.col(m);
	}

	cv::Mat XX = 0 * P;

	for (int j = 0; j < XX.cols; j++)
	{
		XX.col(j) = XX.col(j) + x;
	}

	cv::Mat deltaDiff;
	cv::subtract(P, XX, deltaDiff);
	cv::Mat NN = signum(deltaDiff);
	cv::Mat SNN = cv::Mat::zeros(NN.rows, NN.cols, CV_64F);
	NN.copyTo(SNN);
	cv::Mat UM(SNN.rows, SNN.cols, CV_64F, double(0));

	for (int j = 1; j < SNN.cols; j++)
	{
		SNN.col(j) = NN.col(j) + NN.col(j - 1);
	}

	//cv::Mat II = (cv::Mat(SNN >= 0)).mul(cv::Mat(SNN <= 1));
	//II.convertTo(II, CV_64F);
	//double maxval, minval;
	//cv::minMaxLoc(II, &minval, &maxval);
	//II = II / maxval;

	cv::Mat II(SNN.rows, SNN.cols, CV_64F, double(0));

	II.setTo(1.0, (SNN >= 0) & (SNN <= 1));

	for (double j = 0; j < (double) UM.cols; j++)
	{
		if (((int)j % 2) != 0.0)
		{
			UM.col(j) = x - S.col((int)(j/2));
		}
		else
		{
			A.col((int)(std::floor(j/2))).copyTo(UM.col((int)j));
		}
	}

	UM = UM.mul(II);
	cv::Mat u(UM.rows, 1, CV_64F, double(0));

	for (int m = 0; m < UM.rows; m++)
	{
		u.row(m) = cv::sum(UM.row(m));
	}
	return u;
}

cv::Mat softMSI(const Mat& x, double lambda, double L, const Mat& Wk, const Mat& Z)
{
	cv::Mat A0 = -1e+20 + cv::Mat::zeros(x.rows, 1, CV_64F);
	cv::Mat tempA0 = A0 + 2e+20;
	cv::hconcat(A0, Z, A0);
	cv::hconcat(A0, tempA0, A0);
	cv::Mat A, A_idx;
	cv::sort(A0, A, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING/*CV_SORT_EVERY_COLUMN*/);
	cv::sortIdx(A0, A_idx, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

	cv::Mat tempW0(x.rows, 1, CV_64F, double(0));
	cv::Mat W0(tempW0);
	cv::hconcat(tempW0, Wk, W0);
	cv::hconcat(W0, tempW0, W0);
	cv::Mat W(W0.rows, W0.cols, CV_64F, double(0));

	W0.copyTo(W);

	//cv::Mat w;
	//for (int i = 0; i < x.rows; i++)
	//{
	//	w = (W0.row(i));
	//	for (int j = 0; j < w.cols; j++)
	//	{
	//		int idx = A_idx.at<int>(i, j);
	//		W.at<double>(i, j) = (w.at<double>(0, idx));
	//	}
	//}
	cv::Mat y;
	y = proxMat(x, A, W, lambda, L);
	return y;
}