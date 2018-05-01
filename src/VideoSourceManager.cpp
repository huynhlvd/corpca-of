#include "VideoSourceManager.h"

VideoSourceManager::VideoSourceManager()
{
}

VideoSourceManager::~VideoSourceManager()
{
	vsManager = NULL;
	vSource = NULL;
}

bool VideoSourceManager::instancedFlag = false;
VideoSourceManager* VideoSourceManager::VideoSourceManager::vsManager = NULL;

VideoSourceManager* VideoSourceManager::getInstance()
{
	if (!instancedFlag)
	{
		vsManager = new VideoSourceManager();
		instancedFlag = true;
		return vsManager;
	}
	else
	{
		return vsManager;
	}
}

bool VideoSourceManager::initialize()
{
	vSource = new VideoSource();
	return true;
}

void VideoSourceManager::setDataPath(const std::string datapath)
{
	folderPath = datapath + "\\";
	if (getVideoSource() != NULL)
	{
		getVideoSource()->setPriorPath(folderPath + "prior" + "\\");
	}
}

std::string VideoSourceManager::getDataPath()const
{
	return folderPath;
}

void VideoSourceManager::setOutDataPath(const std::string outPath)
{
	outDataPath = outPath + "\\";
}

std::string VideoSourceManager::getOutDataPath() const
{
	return outDataPath;
}

void VideoSourceManager::selectVideo(int videoSelection)
{
	currSelection = videoSelection;
	std::string videoFile;
	switch (videoSelection)
	{
	case CURTAIN:
		videoFile = "Curtain/*.bmp";
		getVideoSource()->setVideoName("Curtain");
		break;
	case CATS01:
		videoFile = "cats01/*.png";
		getVideoSource()->setVideoName("cats01");
		break;
	case FOUNTAIN:
		videoFile = "Fountain/*.bmp";
		getVideoSource()->setVideoName("Fountain");
		break;
	case HALL:
		videoFile = "hall/*.bmp";
		getVideoSource()->setVideoName("hall");
		break;
	case LABORATORY:
		videoFile = "Laboratory/*.bmp";
		getVideoSource()->setVideoName("Laboratory");
		break;
	case CAMOUFLAGE:
		videoFile = "Camouflage/*.bmp";
		getVideoSource()->setVideoName("Camouflage");
		break;
	case CMUDATA:
		videoFile = "CMUData/*.bmp";
		getVideoSource()->setVideoName("CMUData");
		break;
	case INTERSECTION:
		videoFile = "Intersection/*.jpg";
		getVideoSource()->setVideoName("Intersection");
		break;
	case WAVINGTREES:
		videoFile = "WavingTrees/*.bmp";
		getVideoSource()->setVideoName("WavingTrees");
		break;
	case HIGHWAY:
		videoFile = "highway/*.bmp";
		getVideoSource()->setVideoName("highway");
		break;
	case LIGHTSWITCH:
		videoFile = "LightSwitch/*.bmp";
		getVideoSource()->setVideoName("LightSwitch");
		break;
	case MOVEDOBJECT:
		videoFile = "MovedObject/*.bmp";
		getVideoSource()->setVideoName("MovedObject");
		break;
	case BOOTSTRAP:
	default:
		videoFile = "Bootstrap/*.bmp";
		getVideoSource()->setVideoName("Bootstrap");
		break;
	}
	loadVideo(getDataPath() + videoFile);
}

void VideoSourceManager::selectVideo(const std::string videoname)
{
	seqName =  videoname ;
	getVideoSource()->setVideoName(seqName);
}

void VideoSourceManager::specifyFormat(const std::string formatname)
{
	formatName = formatname;
	std::string fullname = seqName + "/*." + formatName;
	loadVideo(getDataPath() + fullname);
}

int VideoSourceManager::getCurrVideo()
{
	return currSelection;
}

VideoSource* VideoSourceManager::getVideoSource()
{
	return vSource;
}

void VideoSourceManager::loadVideo(std::string videopath)
{
	folderPath = videopath;
	if (getVideoSource() != NULL)
	{
		getVideoSource()->loadVideoFile(videopath);
	}
	extractFileNames(getVideoSource()->getVideoFrameNames());
}

void VideoSourceManager::writeAll(cv::Mat& fg, cv::Mat& bg, cv::Mat& OF1, cv::Mat& OF2, cv::Mat& pfg1, cv::Mat& pfg2, int index)
{
	//write FG
	writeImageToFile("fg_rate_", fg, index);
	//write BG
	writeImageToFile("bg_rate_", bg, index);
	//write OF1
	writeImageToFile("flow12_", OF1, index);
	//write OF2
	writeImageToFile("flow13_", OF2, index);
	//write predicted FG1
	writeImageToFile("predict_fg12_", pfg1, index);
	//write predicted FG2
	writeImageToFile("predict_fg13_", pfg2, index);

}

void VideoSourceManager::writeImageToFile(std::string filename, cv::Mat& img, int index)
{
	std::stringstream ss;
	ss << outDataPath << getVideoSource()->getVideoName() << "/" << filename << setprecision(1) << getVideoSource()->getRate() << "_" << fileNames.at(index);
	cv::Mat imgOut = img;
	if (imgOut.type() != CV_8U)
	{
		imgOut.convertTo(imgOut, CV_8U);
	}
	imwrite(ss.str(), imgOut);
}

void VideoSourceManager::saveCorpcaOFOutput(cv::Mat& fgMat, cv::Mat& bgMat)
{
	std::stringstream ss;
	ss << getOutDataPath() << getVideoSource()->getVideoName() << "/" << "fg_bgCorpcaOF_" << setprecision(1) << getVideoSource()->getRate() << ".yml";

	cv::FileStorage fswrite(ss.str(), cv::FileStorage::WRITE);
	fswrite << "fgCorpcaOF" << fgMat;
	fswrite << "bgCorpcaOF" << bgMat;	
}

std::string VideoSourceManager::appendZeros(int t) const
{
	std::string numZeros = "0";
	int frameCount = vSource->getNumFrames();
	if (t < 10 && t < frameCount)
	{
		if (frameCount >= 1000)
			numZeros = "000";
		else if (frameCount < 1000)
			numZeros = "00";
	}

	else if (t > 9 && t < 100 && t < frameCount)
	{
		if (frameCount >= 1000)
			numZeros = "00";
	}

	else if (t >= 100 && t < frameCount)
	{
		if (frameCount >= 1000)
			numZeros = "0";
	}

	return numZeros;
}

void VideoSourceManager::extractFileNames(std::vector<cv::String> filenames)
{
	//logic to remove full folder path and retain only file names
	for (int t = 0; t < filenames.size(); t++)
	{
		cv::String fullName = filenames.at(t);
		cv::String res = fullName.substr(fullName.find_last_of("\\") + 1);
		fileNames.push_back(res);
	}
}

