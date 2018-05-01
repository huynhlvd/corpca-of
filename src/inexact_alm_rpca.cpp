#include "inexact_alm_rpca.h"
#include "corpcaOFUtils.h"

using namespace corpcaOFUtils;

void inexact_alm_rpca(const cv::Mat &D, double lambda, double tol, int maxIter, cv::Mat& A_hat, cv::Mat& E_hat, int iter)
{
	int m = D.rows;
	int n = D.cols;

	if (tol == -1)
	{
		tol = 1e-7;
	}

	//initialize
	cv::Mat Y;
	D.copyTo(Y);

	//do SVD and get highest value from S --> norm_two
	cv::Mat nU, nW, nVt;
	cv::SVDecomp(Y, nW, nU, nVt, DECOMP_SVD);
	double minval, maxval;
	cv::minMaxLoc(nW, &minval, &maxval);
	double norm_two = maxval;
	double norm_inf = cv::norm(Y, NORM_INF) / lambda;
	double dual_norm = std::max(norm_two, norm_inf);
	Y = Y / dual_norm;

	A_hat = cv::Mat::zeros(m, n, CV_64F);
	E_hat = cv::Mat::zeros(m, n, CV_64F);

	double mu = 1.25 / norm_two;
	double mu_bar = mu * 1e7;
	double rho = 1.5;
	double d_norm = lnorm(D, 2);

	iter = 0;
	int total_svd = 0;
	bool converged = false;
	double stopCriterion = 1;
	int sv = 10;

	while (!converged)
	{
		iter++;

		cv::Mat temp_T = D - A_hat + (1 / mu) * Y;
		E_hat = cv::max(temp_T - lambda / mu, 0);
		E_hat = E_hat + cv::min(temp_T + lambda / mu, 0);

		cv::Mat U, S, V;
		cv::SVDecomp(D - E_hat + (1 / mu) * Y, S, U, V, DECOMP_SVD);

		cv::Mat diagS = cv::Mat::diag(S);

		int svp = cv::countNonZero(diagS > 1 / mu);

		if (svp < sv)
		{
			sv = std::min(svp + 1, n);
		}
		else
		{
			sv = std::min(svp + round(0.05 * n), (double)n);
		}

		cv::Mat tempS = cv::Mat::zeros(svp, 1, CV_64F);

		for (int i = 0; i < svp; i++)
		{
			tempS.at<double>(i, 0) = diagS.at<double>(i, i);
		}
		cv::Mat tempV;
		//transpose(V.colRange(0, svp), tempV);
		A_hat = U.colRange(0, svp) * cv::Mat::diag(tempS) * V.rowRange(0, svp);

		total_svd++;

		cv::Mat Z = D - A_hat - E_hat;

		Y = Y + mu * Z;

		mu = std::min(mu * rho, mu_bar);

		double fnorm = frobNorm(Z);

		stopCriterion = fnorm / dual_norm;

		if (stopCriterion < tol)
		{
			converged = true;
		}
		if (!converged && iter >= maxIter)
		{
			converged = true;
		}
	}

}
