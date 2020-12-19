#pragma once


#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pylon/PylonIncludes.h>
#ifdef PYLON_WIN_BUILD
#include <pylon/PylonGUI.h>
#endif

class PylonCamera
{
private:
	int mSaveImages = 0; // define is images are to be saved, 0-no 1-yes
	int mRecordVideo = 0; // define is videos are to be saved, 0-no, 1-yes
	const int mCountOfImagesToGrab = -1; // number of images to be grabbed


	Pylon::CInstantCamera* mCamera; // pylon camera object
	Pylon::CImageFormatConverter mFormatConverter; // pylon formatConverter
	Pylon::CPylonImage mPylonImage; // pylon image format data

	cv::Size mFrameSize; // opencv frame size of pylon camera

public:
	void PrepareCamera(); // prepare pylon basler camera
	bool GrabCameraFrame(cv::Mat* currFrame); // grab the one frame and return openCV image
	void CloseCamera();

	//void test();
};