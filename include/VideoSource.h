/*
2017
Author: P.V.Srivatsa, srivatsa.pv@live.com
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"

#define SHARE_LIB

#ifdef SHARE_LIB
#define	EXPORT_FUNC __declspec(dllexport)
#else
#define EXPORT_FUNC __declspec(dllimport)
#endif

using namespace std;
using namespace cv;

//Class containing information about a video source
class EXPORT_FUNC VideoSource
{
public:
	//default constructor
	VideoSource();
	//constructor
	VideoSource(std::vector<cv::String> imageFiles);
	//destructor
	~VideoSource();

	//load video file from path
	void loadVideoFile(std::string videopath);

	//initilaize loading
	bool initializeLoading();

	//load prior from file or generate new prior
	void loadPrior(cv::Mat& outB0, cv::Mat& outZ0);

	//load random projection matrix Phi and its inverse from file or generate random matrix
	void loadPhi(cv::Mat& Phi, cv::Mat& invPhi);

	//get frame width
	int getWidth();
	//get frame height
	int getHeight();
	//get number of frames in the video source
	int getNumFrames();

	//set rate m/n
	void setRate(float dataRate = 1.0);

	//get rate
	float getRate();

	//set frame resolution scaling
	void setScale(float scale = 0.5);
	
	//get resolution scale factor
	float getScale();

	//get video file frame names
	std::vector<cv::String> getVideoFrameNames();

	//set path for loading prior
	void setPriorPath(std::string path);

	//get prior path
	std::string getPriorPath();
	//set video source name
	void setVideoName(std::string videoname);
	//get video source name
	std::string getVideoName();

private:
	int width;
	int height;
	size_t numFrames;
	int m;
	int n;
	float RESIZE;
	float rate;
	std::vector<cv::String> fileNames;
	std::string priorPath;
	std::string videoName;
	cv::Mat Phi;
	cv::Mat invPhi;
	cv::Mat B0;
	cv::Mat Z0;
};

