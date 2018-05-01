#include "VideoSource.h"
#include "priorInititialization.h"


VideoSource::VideoSource()
{
}

VideoSource::~VideoSource()
{
}

VideoSource::VideoSource(std::vector<cv::String> imageFiles) : fileNames(imageFiles)
{
	initializeLoading();
}

bool VideoSource::initializeLoading()
{
	cv::Mat img = imread(fileNames.at(0), cv::IMREAD_GRAYSCALE);
	if (img.data != NULL)
	{
		width = (int) ceil(RESIZE * img.cols);
		height = (int) ceil(RESIZE * img.rows);
		if (width % 2 != 0)
		{
			width = (int)floor(RESIZE * img.cols);
		}

		if (height % 2 != 0)
		{
			height = (int)floor(RESIZE * img.rows);
		}

		numFrames = fileNames.size();
		m = (int)ceil(rate * width * height);
		n = (int)ceil(width * height);
		return true;
	}
	else
	{
		return false;
	}
}

void VideoSource::loadVideoFile(std::string videopath)
{
	glob(cv::String(videopath), fileNames);
	initializeLoading();
}

void VideoSource::setRate(float dataRate)
{
	rate = dataRate;
	initializeLoading();
}

void VideoSource::setScale(float scale)
{
	RESIZE = scale;
}

void VideoSource::setPriorPath(std::string path)
{
	priorPath = path;
}

void VideoSource::setVideoName(std::string videoname)
{
	videoName = videoname;
}
std::string VideoSource::getVideoName()
{
	return videoName;
}

float VideoSource::getRate()
{
	return rate;
}

int VideoSource::getWidth()
{
	return width;
}

int VideoSource::getHeight()
{
	return height;
}

int VideoSource::getNumFrames()
{
	return numFrames;
}

float VideoSource::getScale()
{
	return RESIZE;
}

std::vector<cv::String> VideoSource::getVideoFrameNames()
{
	return fileNames;
}

std::string VideoSource::getPriorPath()
{
	return priorPath;
}

void VideoSource::loadPrior(cv::Mat& outB0, cv::Mat& outZ0)
{
	std::stringstream ss;
	ss << priorPath;
	ss << getVideoName() << "\\prior_"  << (int)(1 / RESIZE) << ".yml";

	std::cout << "prior file name: " << ss.str() << std::endl;


	cv::FileStorage fsread(ss.str(), cv::FileStorage::READ);
	if (fsread.isOpened())
	{
		fsread["B0"] >> B0;
		fsread["Z0"] >> Z0;
	}
	else
	{
		priorInitialization(getVideoFrameNames(), 100, 10, RESIZE, B0, Z0, 0);
		cv::FileStorage fswrite(ss.str(), cv::FileStorage::WRITE);
		fswrite << "B0" << B0;
		fswrite << "Z0" << Z0;
	}
	outB0 = B0.clone();
	outZ0 = Z0.clone();
}

void VideoSource::loadPhi(cv::Mat& outPhi, cv::Mat& outinvPhi)
{
	std::stringstream ss;
	ss << priorPath;
	ss << getVideoName() << "\\projMat_" << setprecision(1) << (int)(1/RESIZE) << "_" << rate << ".yml";

	std::cout << "projMat name: " << ss.str() << std::endl;

	if (rate == 1.0)
	{
		Phi = cv::Mat::eye(m, n, CV_64F);
		invPhi = cv::Mat::eye(m, n, CV_64F);
	}
	else
	{
		cv::FileStorage fsread(ss.str(), cv::FileStorage::READ);
		if (fsread.isOpened())
		{
			fsread["Phi"] >> Phi;
			fsread["invPhi"] >> invPhi;

		}
		else
		{
			Phi = cv::Mat::zeros(m, n, CV_64F);
			cv::randn(Phi, cv::Scalar(0.0), Scalar(1.0));
			cv::invert(Phi, invPhi, cv::DECOMP_SVD);
			cv::FileStorage fswrite(ss.str(), cv::FileStorage::WRITE);
			fswrite << "Phi" << Phi;
			fswrite << "invPhi" << invPhi;
		}
		
	}
	outPhi = Phi;
	outinvPhi = invPhi;
}


