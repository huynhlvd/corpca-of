/*
2017
Author: P.V.Srivatsa, srivatsa.pv@live.com
*/

#include "VideoSource.h"

using namespace cv;

//Class to manange video source
class EXPORT_FUNC VideoSourceManager
{
	//singleton class
private:
	static bool instancedFlag;
	static VideoSourceManager *vsManager;
	//constructor
	VideoSourceManager();

public:
	//destructor
	~VideoSourceManager();
	//instantiate video source manager
	static VideoSourceManager* getInstance();
	//load video source from path specified
	void loadVideo(std::string videopath);
	//write all output frames to file
	void writeAll(cv::Mat& fg, cv::Mat& bg, cv::Mat& OF1, cv::Mat& OF2, cv::Mat& pfg1, cv::Mat& pfg2, int index);
	//Append zeros for file names
	std::string appendZeros(int t) const;
	//Write an image to file
	void writeImageToFile(std::string filename, cv::Mat& img, int index);
	//set data path
	void setDataPath(const std::string datapath = "C:/Users/Srivatsa/Documents/Thesis stuff/TRACK/");
	//set output data path
	void setOutDataPath(const std::string outPath = "C:/Users/Srivatsa/Documents/Thesis stuff/TRACK/");
	//get output data path
	std::string getOutDataPath() const;
	//extract file names from the loaded source
	void extractFileNames(std::vector<cv::String> filenames);
	//select video from the list
	void selectVideo(int videoSelection = 1);
	//select video from file name
	void selectVideo(const std::string videoName);
	//specify file format
	void specifyFormat(const std::string formatname);
	//get data path for video sources
	std::string getDataPath()const;
	//get current video
	int getCurrVideo();
	//initialization 
	bool initialize();
	//save CorpcaOF matrices to file
	void saveCorpcaOFOutput(cv::Mat& fgMat, cv::Mat& bgMat);
	//get VideoSource
	VideoSource* getVideoSource();
	//list of video sources available
	enum dataBase {
		BOOTSTRAP = 1,
		CURTAIN = 2,
		FOUNTAIN,
		HALL,
		CATS01,
		CAMOUFLAGE,
		CMUDATA,
		INTERSECTION,
		WAVINGTREES,
		MOVEDOBJECT,
		LIGHTSWITCH,
		HIGHWAY,
		LABORATORY
	};

public:
	std::string folderPath;

	std::string outDataPath;

	std::string seqName;

	std::string formatName;

	int currSelection;

	VideoSource* vSource;

	std::vector<cv::String> fileNames;
};
