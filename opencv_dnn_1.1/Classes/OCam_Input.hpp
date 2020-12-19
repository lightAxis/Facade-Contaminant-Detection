#pragma once

#include <libCamCap.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
//#include <Windows.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


class OCam
{
private:
	const int _CTRL_EXPOSURE = 5;
	const int _CTRL_GAIN = 6;
	const int _CTRL_WB_BLUE = 7;
	const int _CTRL_WB_RED = 8;

	long EXPOSURE = -5;
	long GAIN = 128;
	long WB_BLUE = 216;
	long  WB_RED = 121;

	int IMAGE_WIDTH = 640;
	int IMAGE_HEIGHT = 480;
	int IMAGE_FPS = 60;

	const char* mWindowName = "oCam";

	cv::Mat mRawframe;
	cv::Mat mFrame;

	int mCamNum;
	char* mCamModel;
	CAMPTR mPtrCam;

	bool mbCameraRunning = false;

public:
	enum eWIDTH_HEIGHT { W640_H480 };
	enum eFPS { FPS60 };

	OCam(const eWIDTH_HEIGHT& Width_Height, const eFPS& Fps);
	~OCam();
	void SetCameraParameters(const long& exposure, const long& gain, const long& WB_Blue, const long& WB_Red);

	bool StartOCam();
	bool CloseOCam();

	cv::Mat GetFrame();

	bool IsRunningNow();
};