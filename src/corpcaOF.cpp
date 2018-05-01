#include "corpcaOF.h"
#include "corpcaOFUtils.h"
#include "ramsia.h"

using namespace cv;
#ifdef CUDA
using namespace cv::cuda;
#endif
using namespace corpcaOFUtils;

void incSVD(const cv::Mat& v, const cv::Mat& U0, const cv::Mat& S0, const cv::Mat& V0, cv::Mat& U1, cv::Mat& S1, cv::Mat& V1)
{
	int Ncols = U0.cols;
	cv::Mat transposedU0;
	cv::transpose(U0, transposedU0);
	cv::Mat r = transposedU0 * v;
	cv::Mat tempz = U0 * r;
	cv::Mat z = v - tempz; // tempz;
	double rho = sqrt(cv::sum(z.mul(z))[0]);
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
	//cv::transpose(Gv, Gv);
	S1 = cv::Mat::diag(S1);
	cv::Mat V0_t;
	cv::transpose(V0, V0_t);
	//cv::Mat U1, V1;
	cv::hconcat(U0, p, U1);
	cv::transpose(Gv_t, Gv);
	U1 = U1 * Gu;
	cv::hconcat(V0_t, cv::Mat::zeros(max(V0_t.rows, V0_t.cols), 1, CV_64F), V1);
	cv::Mat tempV1;
	cv::hconcat(cv::Mat::zeros(1, Ncols, CV_64F), cv::Mat(1, 1, CV_64F, 1.0), tempV1);
	cv::vconcat(V1, tempV1, V1);
	V1 = V1 * Gv;
}

void corpca(const cv::Mat& yt, const cv::Mat& Phi, const cv::Mat& invPhi, cv::Mat& Ztm1, cv::Mat& Btm1, cv::Mat& xt, cv::Mat& vt, cv::Mat& Zt, cv::Mat& Bt)
{
	//Input
	int m, n;
	m = Phi.rows;
	n = Phi.cols;
	float lambda0 = 1 / sqrt(n);
	cv::Mat c, G;
	auto totstart = get_time::now();
	if (m == n)
	{
		c = yt;
		G = invPhi;
	}
	else
	{
		c = invPhi * yt;
		G = invPhi * Phi;
	}

	//Initialize optimization variables
	std::vector<double> lamdaAdapt;
	lamdaAdapt.push_back(lambda0);
	//lamdaAdapt.push_back(2 * lambda0);
	lamdaAdapt.push_back(3 * lambda0);
	cv::Mat Btout;

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

		

		cv::Mat u0, s0, v0;
		
		auto svdstart = get_time::now();
		//computeSVD(Btm1, s0, u0, v0);
		cv::SVDecomp(Btm1, S0, U0, V0);

		auto svdend = get_time::now();
		auto svdtime = svdend - svdstart;
		std::cout << "SVD time: " << chrono::duration_cast<ms>(svdtime).count() << "ms" << std::endl;

		S0 = cv::Mat::diag(S0);
		cv::Mat vt_kp1, xt_kp1;
		auto iterstart = get_time::now();
		while (!converged)
		{
			cv::Mat tvt_k = vt_k + ((sai_km1 - 1) / sai_k) * (vt_k - vt_km1);
			cv::Mat txt_k = xt_k + ((sai_km1 - 1) / sai_k) * (xt_k - xt_km1);

			cv::Mat thvt_k = tvt_k - (1 / tau_k) * (G * (tvt_k + txt_k) - c);
			cv::Mat thxt_k = txt_k - (1 / tau_k) * (G * (tvt_k + txt_k) - c);

			//std::cout << "*************iter: " << numIter << std::endl;

			auto incSVDstart = get_time::now();

			incSVD(thvt_k, U0, S0, V0, U1, S1, V1);

			auto incSVDend = get_time::now();
			auto incSVDTime = incSVDend - incSVDstart;
			//std::cout << "incSVD time: " << chrono::duration_cast<ms>(incSVDTime).count() << "ms" << std::endl;

			cv::Mat diagS = cv::Mat::zeros(S1.rows, 1, CV_64F);
			(S1.diag()).copyTo(diagS);
			cv::Mat tempS = diagS - mu_k / tau_k;

			cv::Mat mask = tempS > 0;
			mask.convertTo(mask, CV_64F);
			mask = mask / 255.0;
			proxS = tempS.mul(mask);
			proxS = cv::Mat::diag(proxS);

			cv::transpose(V1, V1_t);
			cv::Mat Tht = U1 * proxS * V1_t;
			vt_kp1 = Tht.col(Tht.cols - 1);

			auto sofmsistart = get_time::now();
			xt_kp1 = softMSI(thxt_k, mu_k * lambda, tau_k, Wk, Z);
			auto softmsiend = get_time::now();
			auto softmsitime = softmsiend - sofmsistart;
			//std::cout << "softMSI time: " << chrono::duration_cast<ms>(softmsitime).count() << "ms" << std::endl;

			auto wtime = get_time::now();
			/*update weights of RAMSIA*/
			for (int j = 0; j < J; j++)
			{
				cv::Mat denom = epsi + cv::abs(xt_kp1 - Z.col(j));
				cv::divide(1, denom, Wkp1.col(j));
				//Wkp1.col(j) = 1./
				double tempsum = bigN / (cv::sum(Wkp1.col(j))[0]);
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
			auto wstop = get_time::now() - wtime;
			//std::cout << "Weigth update: " << numIter << " :   "  << chrono::duration_cast<ms>(wstop).count() << "ms"  << std::endl;
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
		auto iterend = get_time::now();
		auto itertime = iterend - iterstart;
		std::cout << "Num iterations: " << numIter << std::endl;
		std::cout << "Convergence time: " << l << ": " << chrono::duration_cast<ms>(itertime).count() << "ms" << std::endl;
		std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

		auto tstart = get_time::now();
		vt = vt_kp1;
		xt = xt_kp1;

		//Updating prior information for the next instance
		Zt = Ztm1;
		Ztm1.colRange(1, J - 2).copyTo(Zt.colRange(0, J - 3));

		xt.copyTo(Zt.col(J - 2));

		cv::Mat res = U1.colRange(0, Btm1.cols) * proxS(Range(0, Btm1.cols), Range(0, Btm1.cols)) * V1(Range(0, Btm1.cols), Range(0, Btm1.cols));
		res.copyTo(Bt);
		Bt.copyTo(Btm1);

		if (l == 0)
		{
			Bt.copyTo(Btout);
		}

		auto tend = get_time::now() - tstart;
		std::cout << "CORPCA Copy time: "  << chrono::duration_cast<ms>(tend).count() << "ms" << std::endl;
	}
	Btout.copyTo(Bt);
	auto totaltime = get_time::now() - totstart;
	std::cout << "Total CORPCA time: " << chrono::duration_cast<ms>(totaltime).count() << "ms" << std::endl;
}

void corpcaOF(VideoSourceManager* vsManager, cv::Mat& fgCorpca, cv::Mat& bgCorpca)
{
	//initialize all variables
	int nSI = 3;
	VideoSource* vSource = vsManager->getVideoSource();
	cv::Mat Phi, invPhi;
	cv::Mat B0, Z0;

	vSource->loadPhi(Phi, invPhi);
	vSource->loadPrior(B0, Z0);

	cv::Mat Btm1 = B0;
	cv::Mat Ztm1 = cv::Mat::zeros(Btm1.rows, nSI, CV_64F);

	std::vector<cv::String> fileNames = vSource->getVideoFrameNames();

	fgCorpca = cv::Mat::zeros(Phi.cols, fileNames.size(), CV_64F);
	bgCorpca = cv::Mat::zeros(Phi.cols, fileNames.size(), CV_64F);

	int startFrame = 0;
	int endFrame = vSource->getNumFrames();
	std::cout << "Sequence: " << vSource->getVideoName() << std::endl;;
	std::cout << "Used resolution: " << vSource->getWidth() << " x " << vSource->getHeight() << std::endl;


	//start loop for CORPCA and OF
	std::cout << std::endl << "Starting loop" << std::endl;

	for (int t = startFrame; t < endFrame; t++)
	{
		cv::Mat xt, vt, Zt, Bt;
		cv::Mat myImage, currFrame;

		std::cout << "Testing frame " << t << " at a measurement rate " << vSource->getRate() << std::endl;
		auto starttime = get_time::now();
		//read video frame
		myImage = imread(fileNames.at(t), cv::IMREAD_GRAYSCALE);

		//resize the frame
		cv::resize(myImage, currFrame, Size(), vSource->getScale(), vSource->getScale(), cv::INTER_LANCZOS4);

		//convert into a column matrix
		currFrame = currFrame.reshape(0, 1);
		currFrame.convertTo(currFrame, CV_64F);
		cv::transpose(currFrame, currFrame);
		//multiplying with projection matrix
		cv::Mat yt = Phi * currFrame;

		auto corpcastart = get_time::now();

		//xt, vt, Zt, Bt are outputs of CORPCA
		corpca(yt, Phi, invPhi, Ztm1, Btm1, xt, vt, Zt, Bt);

		auto corpcatime = get_time::now() - corpcastart;
		std::cout << "CORPCA time: " << chrono::duration_cast<ms>(corpcatime).count() << "ms" << std::endl;

		//threshold xt
		double minx, maxx;
		cv::minMaxLoc(abs(xt + vt), &minx, &maxx);
		double noise_thres = 1 * abs(minx);
		xt.setTo(0, abs(xt) < noise_thres);

		//save separated fg and bg
		xt.copyTo(fgCorpca.col(t));
		vt.copyTo(bgCorpca.col(t));

		//to save fg and bg to file
		cv::Mat fg_img = xt;
		if (!fg_img.isContinuous())
			fg_img = fg_img.clone();
		fg_img = fg_img.reshape(0, vSource->getHeight());
		fg_img = scaledData(fg_img, 0, 255);

		cv::Mat bg_img = vt;
		if (!bg_img.isContinuous())
			bg_img = bg_img.clone();
		bg_img = bg_img.reshape(0, vSource->getHeight());
		bg_img = scaledData(bg_img, 0, 255);

		if (t < startFrame + 2)
		{
			vsManager->writeImageToFile("fg_rate_", fg_img, t);
			vsManager->writeImageToFile("bg_rate_", bg_img, t);
		}

		//update prior information
		Ztm1 = Zt;
		Btm1 = Bt;

		//OF to improve prior information
		if (t >= startFrame + 2 && t < endFrame)
		{
			auto ofstart = get_time::now();
			cv::Mat x_t, x_tm1, x_tm2;
			//reshape 3 latest priors from column matrix to image size
			convertMatToImage(fgCorpca.col(t), x_t, vSource->getHeight());
			convertMatToImage(fgCorpca.col(t - 1), x_tm1, vSource->getHeight());
			convertMatToImage(fgCorpca.col(t - 2), x_tm2, vSource->getHeight());

			//motion estimation
			//std::stringstream ss;
			//ss << "test.yml";
			//cv::FileStorage fswrite(ss.str(), cv::FileStorage::WRITE);
			//fswrite << "x_t" << x_t;
			//fswrite << "x_tm1" << x_tm1;
			//fswrite << "x_tm2" << x_tm2;
			
			cv::Mat flowX01, flowY01, outOF01;
			//compute OF between frames x(t) and x(t-1)
			computeOF(x_t, x_tm1, flowX01, flowY01, outOF01);

			cv::Mat flowX02, flowY02, outOF02;
			//compute OF between frames x(t) and x(t-2)
			computeOF(x_t, x_tm2, flowX02, flowY02, outOF02);

			//computing horizontal and vertical indidces for the flow vectors
			cv::Mat ind01X, ind01Y;
			computeFlowIndices(flowX01, flowY01, vSource->getWidth(), vSource->getHeight(), 1.0, ind01X, ind01Y);
			cv::Mat ind02X, ind02Y;
			computeFlowIndices(flowX02, flowY02, vSource->getWidth(), vSource->getHeight(), 0.5, ind02X, ind02Y);

			//motion compensation using computed motion vectors
			cv::Mat fg0 = linearMotionCompensation(x_t, ind01X, ind01Y);
			cv::Mat fg1 = linearMotionCompensation(x_t, ind02X, ind02Y);
			auto ofstop = get_time::now();
			auto oftime = ofstop - ofstart;
			std::cout << "OF time: " << chrono::duration_cast<ms>(oftime).count() << "ms" << std::endl;

			auto tstart = get_time::now();

			//scaling image to range 0 to 255
			cv::Mat fg0_img = scaledData(fg0, 0, 255);
			cv::Mat fg1_img = scaledData(fg1, 0, 255);

			//save all images to file
			vsManager->writeAll(fg_img, bg_img, outOF01, outOF02, fg0_img, fg1_img, t);


			//update Zt
			cv::Mat prior0 = cv::Mat::zeros(fg0.cols * fg0.rows, 1, CV_64F);
			for (int i = 0; i < fg0.cols; i++)
			{
				int rowIndex = i * fg0.rows;
				fg0.col(i).copyTo(prior0.rowRange(rowIndex, rowIndex + fg0.rows));
			}

			prior0.copyTo(Zt.col(1));

			cv::Mat prior1 = cv::Mat::zeros(fg1.cols * fg1.rows, 1, CV_64F);
			for (int i = 0; i < fg0.cols; i++)
			{
				int rowIndex = i * fg1.rows;
				fg1.col(i).copyTo(prior1.rowRange(rowIndex, rowIndex + fg1.rows));
			}

			prior1.copyTo(Zt.col(0));
			Ztm1 = Zt;

			auto tend = get_time::now() - tstart;
			std::cout << "CORPCA-OF Copy time: " << chrono::duration_cast<ms>(tend).count() << "ms" << std::endl;
		}

		auto stoptime = get_time::now();
		auto diff = stoptime - starttime;
		std::cout << "*******************CORPCA-OF time: " << chrono::duration_cast<ms>(diff).count() << "ms************************" << std::endl << std::endl;
	}
}

void convertMatToImage(cv::Mat& mat, cv::Mat& img, int numRows)
{
	img = mat.clone();
	img = img.reshape(0, numRows);
}

#ifdef CUDA
void computeOF(cv::Mat& x1, cv::Mat& x2, cv::Mat& flowX, cv::Mat& flowY, cv::Mat& outOF)
{
	//initialize device side(GPU) matrices
	cv::cuda::GpuMat d_x1(x1), d_x2(x2);
	cv::cuda::GpuMat d_OF(x1.size(), CV_32FC2);

	//convert GpU matrices to float type
	cv::cuda::GpuMat d_x1_f, d_x2_f;
	d_x1.convertTo(d_x1_f, CV_32F, 1.0 / 255.0);
	d_x2.convertTo(d_x2_f, CV_32F, 1.0 / 255.0);

	//initilaize broxLDOF
	Ptr<cuda::BroxOpticalFlow> broxLDOF = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

	broxLDOF->calc(d_x1_f, d_x2_f, d_OF);

	GpuMat planes01[2];
	cuda::split(d_OF, planes01);

	cv::Mat flowx(planes01[0]);
	cv::Mat flowy(planes01[1]);

	flowX = flowx;
	flowY = flowy;
	outOF = writeFlow(d_OF);
}

#else
void computeOF(cv::Mat& x1, cv::Mat& x2, cv::Mat& flowX, cv::Mat& flowY, cv::Mat& outOF)
{
	//initialize device side(GPU) matrices
	cv::Mat d_x1(x1), d_x2(x2);
	cv::Mat d_x1_f, d_x2_f;
	cv::Mat d_OF(x1.size(), CV_32FC2);
	//cv::normalize(d_x1, d_x1);
	//cv::normalize(d_x2, d_x2);
	//convert GpU matrices to float type
	d_x1.convertTo(x1, CV_32F, 1.0 / 255.0);
	d_x2.convertTo(x2, CV_32F, 1.0 / 255.0);
	Ptr<DualTVL1OpticalFlow> tvl1 = createOptFlow_DualTVL1();
	//tvl1->setMedianFiltering(4);
	//initilaize broxLDOF
	tvl1->calc(x1, x2, d_OF);
	//calcOpticalFlowFarneback(d_x1, d_x2, d_OF, 0.5, 3, 15, 3, 5, 1.2, 0);
	
	cv::Mat planes01[2];
	cv::split(d_OF, planes01);

	cv::Mat flowx(planes01[0]);
	cv::Mat flowy(planes01[1]);

	flowX = flowx;
	flowY = flowy;
	outOF = writeFlow(d_OF);
}
#endif




void computeFlowIndices(cv::Mat& flowx, cv::Mat& flowy, int width, int height, float mulFactor, cv::Mat& indX, cv::Mat& indY)
{
	//compute mesh grid
	cv::Mat XX, YY;
	meshgrid(cv::Range(0, width - 1), cv::Range(0, height - 1), XX, YY);

	indX = XX + mulFactor * flowx;
	indX.setTo(0, indX < 0);
	indX.setTo(width - 1, indX > width - 1);

	indY = YY + mulFactor * flowy;
	indY.setTo(0, indY < 0);
	indY.setTo(height - 1, indY > height - 1);
}
