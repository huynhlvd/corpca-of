#include <iostream>
#include <chrono>

#include "corpcaOFUtils.h"
#include "corpcaOF.h"


using namespace corpcaOFUtils;

int main()
{
	cv::Mat B0, Z0;
	cv::Mat Phi, invPhi;

	VideoSourceManager* vsManager = VideoSourceManager::getInstance();
	vsManager->initialize();
	std::cout << "Enter path to the dataset: ";
	std::string folderpath;
	std::cin >> folderpath;
	//enter the path here
	vsManager->setDataPath(folderpath);
	std::cout << "\nEnter path to save output files: ";
	std::string outpath;
	std::cin >> outpath;
	vsManager->setOutDataPath(outpath);
	std::cout << "\nSelect the video sequence: ";
	std::string videoName;
	std::cin >> videoName;
	//vsManager->selectVideo(VideoSourceManager::BOOTSTRAP);
	vsManager->selectVideo(videoName);
	std::cout << "\nSpecify the sequence format type (bmp, png, tiff etc.) : ";
	std::string fomatName;
	std::cin >> fomatName;
	vsManager->specifyFormat(fomatName);
	VideoSource* vSource = vsManager->getVideoSource();

	float rate = 1.0, scale = 0.5;
	std::cout << "Enter scale factor :";
	std::cin >> scale;
	
	std::cout << "\nEnter rate :";
	std::cin >> rate;
	vSource->setScale(scale);
	vSource->setRate(rate);
	cv::Mat fgCorpcaOF, bgCorpcaOF;
	corpcaOF(vsManager, fgCorpcaOF, bgCorpcaOF);
	vsManager->saveCorpcaOFOutput(fgCorpcaOF, bgCorpcaOF);

	std::cout << "CORPCA-OF done" << std::endl;
	

	return 0;
}
