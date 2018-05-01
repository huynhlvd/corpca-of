#include "corpca.h"

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

 double frobNorm(cv::Mat& M)
{
	cv::Mat M_t;
	cv::transpose(M, M_t);
	double sum = cv::trace(M_t * M)[0];

	double norm = std::sqrt(sum);
	return norm;
}

 void incSVD(const cv::Mat& v, const cv::Mat& U0, const cv::Mat& S0, const cv::Mat& V0, cv::Mat& U1, cv::Mat& S1, cv::Mat& V1)
{
	int Ncols = U0.cols;
	cv::Mat transposedU0;
	cv::transpose(U0, transposedU0);
	cv::Mat r = transposedU0 * v;
	cv::Mat tempz = U0 * r;
	cv::Mat z = v - tempz;
	double rho = sqrt(sum(z.mul(z))[0]);
	cv::Mat p;
	if (rho > 1e-8)
	{
		p = z / rho;
	}
	else
	{
		p = cv::Mat::zeros(z.rows, z.cols, CV_64F);
	}

	cv::Mat St, tempSt1, tempSt2;
	cv::hconcat(S0, r, tempSt1);
	cv::hconcat(cv::Mat::zeros(1, Ncols, CV_64F), cv::Mat(1, 1, CV_64F, rho), tempSt2);
	cv::vconcat(tempSt1, tempSt2, St);

	cv::Mat Gu, Gv, Gv_t;
	cv::SVDecomp(St, S1, Gu, Gv_t);
	S1 = cv::Mat::diag(S1);
	cv::Mat V0_t;
	cv::transpose(V0, V0_t);
	cv::hconcat(U0, p, U1);
	cv::transpose(Gv_t, Gv);
	U1 = U1 * Gu;
	cv::hconcat(V0_t, cv::Mat::zeros(max(V0_t.rows, V0_t.cols), 1, CV_64F), V1);
	cv::Mat tempV1;
	cv::hconcat(cv::Mat::zeros(1, Ncols, CV_64F), cv::Mat(1, 1, CV_64F, 1.0), tempV1);
	cv::vconcat(V1, tempV1, V1);
	V1 = V1 * Gv;
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
	cv::Mat w;
	for (int i = 0; i < x.rows; i++)
	{
		w = (W0.row(i));
		for (int j = 0; j < w.cols; j++)
		{
			int idx = A_idx.at<int>(i, j);
			W.at<double>(i, j) = (w.at<double>(0, idx));
		}
	}

	cv::Mat y;
	y = proxMat(x, A, W, lambda, L);
	return y;

}

//signum function to return sign if different and 0 if same
 cv::Mat signum(cv::Mat src)
{
	cv::Mat z = Mat::zeros(src.size(), src.type());
	cv::Mat a = (z < src) & 1;
	cv::Mat b = (src < z) & 1;

	Mat dst;
	addWeighted(a, 1.0, b, -1.0, 0.0, dst, CV_64F);
	return dst;

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

	for (int j = 0; j < (2 * A.cols - 2); j++)
	{
		XX.col(j) = XX.col(j) + x;
	}

	cv::Mat deltaDiff;
	cv::subtract(P, XX, deltaDiff);
	cv::Mat NN = signum(deltaDiff);
	cv::Mat SNN = cv::Mat::zeros(NN.rows, NN.cols, CV_64F);
	NN.copyTo(SNN);
	cv::Mat UM(SNN.rows, SNN.cols, CV_64F, double(0));

	for (int j = 1; j < (2 * (A.cols - 1)); j++)
	{
		SNN.col(j) = NN.col(j) + NN.col(j - 1);
	}
	cv::Mat II = (cv::Mat(SNN >= 0)).mul(cv::Mat(SNN <= 1));
	II.convertTo(II, CV_64F);
	double maxval, minval;
	cv::minMaxLoc(II, &minval, &maxval);
	II = II / maxval;

	for (int j = 0; j < (2 * (A.cols - 1)); j++)
	{
		if ((j % 2) != 0)
		{
			double ind = (double)j / 2;
			UM.col(j) = x - S.col(ind);
		}
		else
		{
			double ind = std::floor((double)((double)j / 2));
			A.col(ind).copyTo(UM.col(j));
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

 void corpca(const cv::Mat& yt, const cv::Mat& Phi, const cv::Mat& invPhi, const cv::Mat& Ztm1, cv::Mat& Btm1, cv::Mat& xt, cv::Mat& vt, cv::Mat& Zt, cv::Mat& Bt)
{
	//	//Input
	int m, n;
	m = Phi.rows;
	n = Phi.cols;
	float lambda0 = 1 / sqrt(n);
	cv::Mat c, G;
	if (m == n)
	{
		c = yt;
	}
	else
	{
		//cv::invert(Phi, G, cv::DECOMP_SVD);
		c = invPhi * yt;
		G = invPhi * Phi;
	}
	////used for parameter counting - not needed
	//	cv::Mat lambdaAdapt(1, 2, CV_64F);
	//	lambdaAdapt.col(0) = lambda0;
	//	lambdaAdapt.col(1) = 3 * lambda0;
	//Initialize optimization variables
	//int m, n;
	//m = Phi.rows;
	//n = Phi.cols;
	//float lambda0 = 1 / sqrt(n);
	std::vector<double> lamdaAdapt;
	lamdaAdapt.push_back(lambda0);
	lamdaAdapt.push_back(3 * lambda0);
	for (int l = 0; l < lamdaAdapt.size(); l++)
	{
		double lambda = lamdaAdapt.at(l);

		double sai_k = 1; //t^k
		double sai_km1 = 1; //t^(k-1) 
		int tau_0 = 2;
		int maxIter = 1000;
		cv::Mat vt_km1, vt_k;
		cv::reduce(Btm1.colRange(0, Btm1.cols), vt_km1, 1, CV_REDUCE_AVG);
		vt_km1.copyTo(vt_k);

		cv::Mat xt_km1 = cv::Mat::zeros(n, 1, CV_64F); //X^{k-1} = (A^{k-1},E^{k-1})
		cv::Mat xt_k = cv::Mat::zeros(n, 1, CV_64F); //X^{k} = (A^{k},E^{k})

													 //Input SIs
		int J = Ztm1.cols + 1;
		cv::Mat Wk = cv::Mat::zeros(n, J, CV_64F); // Weights on source
		cv::Mat beta = cv::Mat::zeros(J, 1, CV_64F);
		beta.row(0) = 1;
		Wk.col(0) = 1;
		cv::Mat Z = cv::Mat::zeros(n, J, CV_64F);
		Ztm1.copyTo(Z.colRange(1, Z.cols));

		double mu_0 = cv::norm(c);
		double mu_k = 0.99 * mu_0;
		double mu_bar = 1e-9 * mu_0;

		double tau_k = tau_0;

		bool converged = false;
		bool continuationFlag = true;
		unsigned int numIter = 0;

		double epsi = 0.8; //obtained via experiments
		double epsiBeta = 1e-20;
		int bigN = n;
		cv::Mat Wkp1;
		Wk.copyTo(Wkp1);

		//start main loop
		cv::Mat U0, S0, V0;
		cv::Mat U1, S1, V1;
		cv::Mat V1_t;
		cv::Mat proxS;

		cv::SVDecomp(Btm1, S0, U0, V0);

		S0 = cv::Mat::diag(S0);
		cv::Mat vt_kp1, xt_kp1;
		while (!converged)
		{
			cv::Mat tvt_k = vt_k + ((sai_km1 - 1) / sai_k) * (vt_k - vt_km1);
			cv::Mat txt_k = xt_k + ((sai_km1 - 1) / sai_k) * (xt_k - xt_km1);

			cv::Mat thvt_k = tvt_k - (1 / tau_k) * (G * (tvt_k + txt_k) - c);
			cv::Mat thxt_k = txt_k - (1 / tau_k) * (G * (tvt_k + txt_k) - c);


			incSVD(thvt_k, U0, S0, V0, U1, S1, V1);
			cv::Mat diagS = cv::Mat::zeros(S1.rows, 1, CV_64F);
			(S1.diag()).copyTo(diagS);
			cv::Mat tempS = diagS - mu_k / tau_k;

			cv::Mat mask = tempS > 0;
			mask.convertTo(mask, CV_64F);
			mask = mask / 255.0;
			proxS = tempS.mul(mask);
			proxS = cv::Mat::diag(proxS);

			//cv::transpose(V1, V1);

			cv::transpose(V1, V1_t);
			cv::Mat Tht = U1 * proxS * V1_t;
			vt_kp1 = Tht.col(Tht.cols - 1);
			xt_kp1 = softMSI(thxt_k, mu_k * lambda, tau_k, Wk, Z);

			/*update weights of RAMSIA*/
			for (int j = 0; j < J; j++)
			{
				cv::Mat denom = epsi + cv::abs(xt_kp1 - Z.col(j));
				cv::divide(1, denom, Wkp1.col(j));

				double tempsum = bigN / (sum(Wkp1.col(j))[0]);
				Wkp1.col(j) = Wkp1.col(j) * tempsum;
			}

			for (int j = 0; j < J; j++)
			{
				double denom_scalar = 0;
				for (int i = 0; i < J; i++)
				{
					double numr = epsiBeta + cv::norm(Wkp1.col(j).mul(xt_kp1 - Z.col(j)), cv::NORM_L1);
					double denr = epsiBeta + cv::norm(Wkp1.col(i).mul(xt_kp1 - Z.col(i)), cv::NORM_L1);

					denom_scalar += numr / denr;
				}
				beta.at<double>(j, 0) = 1 / denom_scalar;
			}

			for (int j = 0; j < J; j++)
			{
				Wkp1.col(j) = Wkp1.col(j) * beta.row(j);
			}

			double sai_kp1 = 0.5 * (1 + sqrt(1 + 4 * sai_k * sai_k));

			cv::Mat tempSum = vt_kp1 + xt_kp1 - tvt_k - txt_k;
			cv::Mat S_kp1_vt = tau_k * (tvt_k - vt_kp1) + tempSum;
			cv::Mat S_kp1_xt = tau_k * (txt_k - xt_kp1) + tempSum;

			cv::Mat S_kp1;
			cv::hconcat(S_kp1_vt, S_kp1_xt, S_kp1);
			cv::Mat vtxt;
			cv::hconcat(vt_kp1, xt_kp1, vtxt);
			double stoppingCriterion = frobNorm(S_kp1) / (tau_k * std::max(1.0, frobNorm(vtxt)));

			if (stoppingCriterion <= 5 * 1e-6)
				converged = true;

			if (continuationFlag)
				mu_k = std::max(0.9 * mu_k, mu_bar);

			sai_km1 = sai_k;
			sai_k = sai_kp1;

			vt_km1 = vt_k;
			xt_km1 = xt_k;

			vt_k = vt_kp1;
			xt_k = xt_kp1;

			Wk = Wkp1;

			numIter++;

			if (!converged && numIter >= maxIter)
				converged = true;

		}

		vt = vt_kp1;
		xt = xt_kp1;

		//Updating prior information for the next instance
		Zt = Ztm1;
		Ztm1.colRange(0, J - 3).copyTo(Zt.colRange(1, J - 2));

		xt.copyTo(Zt.col(J - 2));

		cv::Mat res = U1.colRange(0, Btm1.cols) * proxS(Range(0, Btm1.cols), Range(0, Btm1.cols)) * V1(Range(0, Btm1.cols), Range(0, Btm1.cols));
		res.copyTo(Bt);
		Bt.copyTo(Btm1);
	}
}
