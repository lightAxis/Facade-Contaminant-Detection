#pragma once

#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include "InBoxChecker_Tool.hpp"
#include "YOLO_v3_Module.hpp"
#include "ColorDetection_Module.hpp"
#include "GrayScale_Module.hpp"

class RobustOptimalExperiment
{
public:
	enum eOrthogonalArray { L9 = 9, L27 = 27, L_None = 0 };
private:
	float mNMSThreshold;
	float mConfidenceThreshold;

	int mHueMargin;
	int mMedianFilterSize;
	int mDownsamplingSize;

	GrayScaleCalculator::eCalcMethod mGrayscaleMethod;

	const int mL9_Array[9][4] = {
		{1,1,1,1},//1
		{1,2,2,2},//2
		{1,3,3,3},//3
		{2,1,2,3},//4
		{2,2,3,1},//5
		{2,3,1,2},//6
		{3,1,3,2},//7
		{3,2,1,3},//8
		{3,3,2,1} //9
	};

	const int mL27_Array[27][13] = {
		{1,1,1,1,1,1,1,1,1,1,1,1,1},//1
		{1,1,1,1,2,2,2,2,2,2,2,2,2},//2
		{1,1,1,1,3,3,3,3,3,3,3,3,3},//3
		{1,2,2,2,1,1,1,2,2,2,3,3,3},//4
		{1,2,2,2,2,2,2,3,3,3,1,1,1},//5
		{1,2,2,2,3,3,3,1,1,1,2,2,2},//6
		{1,3,3,3,1,1,1,3,3,3,2,2,2},//7
		{1,3,3,3,2,2,2,1,1,1,3,3,3},//8
		{1,3,3,3,3,3,3,2,2,2,1,1,1},//9
		{2,1,2,3,1,2,3,1,2,3,1,2,3},//10
		{2,1,2,3,2,3,1,2,3,1,2,3,1},//11
		{2,1,2,3,3,1,2,3,1,2,3,1,2},//12
		{2,2,3,1,1,2,3,2,3,1,3,1,2},//13
		{2,2,3,1,2,3,1,3,1,2,1,2,3},//14
		{2,2,3,1,3,1,2,1,2,3,2,3,1},//15
		{2,3,1,2,1,2,3,3,1,2,2,3,1},//16
		{2,3,1,2,2,3,1,1,2,3,3,1,2},//17
		{2,3,1,2,3,1,2,2,3,1,1,2,3},//18
		{3,1,3,2,1,3,2,1,3,2,1,3,2},//19
		{3,1,3,2,2,1,3,2,1,3,2,1,3},//20
		{3,1,3,2,3,2,1,3,2,1,3,2,1},//21
		{3,2,1,3,1,3,2,2,1,3,3,2,1},//22
		{3,2,1,3,2,1,3,3,2,1,1,3,2},//23
		{3,2,1,3,3,2,1,1,3,2,2,1,3},//24
		{3,3,2,1,1,3,2,3,2,1,2,1,3},//25
		{3,3,2,1,2,1,3,1,3,2,3,2,1},//26
		{3,3,2,1,3,2,1,2,1,3,1,3,2},//27
	};

	std::vector<std::string> getFilenames(const std::string& folderAddress, std::vector<std::string>* fileNames);
	std::vector<cv::Rect> getYOLOBoxes(const std::string& path, const int& frameCols, const int& frameRows);
	float compare_YOLO_IoUc(const std::vector<cv::Rect>& answerBoxes, const std::vector<cv::Rect>& detectedBoxes,
		const int& frameCols, const int& frameRows);

	float getColorSectionAreaRatioAnswer(const std::string& path);
	std::vector<cv::Rect> getColorBoxes(const std::string& path, const int& FrameCols, const int& FrameRows);
	std::vector<cv::Rect> getYOLOv3DetectionBox(const std::string& imgAdded, InBoxChecker* InboxCheckerTool);
	std::vector<cv::Rect> getColorDetectionDetectionBox(const std::string& imgAdded, InBoxChecker* inboxCheckerTool);

	float compareColorDetection_errIoUa(const std::vector<cv::Rect>& answerBoxes, InBoxChecker* currentInboxChecker,
		const float& answerAreaRatio, const float& detectedAreaRatio, const long long& elapsedTime, const int& frameCols, const int& frameRows);
	std::vector<std::string> splitStringByChar(const std::string& str, const char& delimiter);

public:
	void DoYoloExperiment(YOLO_v3_DNN* YOLOv3_Module,
		const float(&NMSThresholdLevels)[], const float(&ConfidenceThresholdLevels)[],
		const eOrthogonalArray& orthArray = L9);

	void DoColorDetection_Experiment(YOLO_v3_DNN* YOLOv3_Module, const float& fixedNMSThreshold, const float& fixedConfidenceThreshold,
		ColorDetection* colorDetection_Module, const int(&hueMarginLevels)[], const int(&medianFilterSizeLevels)[], const int(&downsamplingSizeLevels)[],
		const eOrthogonalArray& orthArray = L27);

	void DoGrayScaleExperiment(YOLO_v3_DNN* YOLOv3_Module, const float& fixedNMSThreshold, const float& fixedConfidenceThreshold,
		ColorDetection* ColorDetection_Module, const int& fixedHueMargin, const int& fixedMedianFilterSize, const int& fixedDownsamplingSize,
		GrayScaleCalculator* GrayScale_Module, const GrayScaleCalculator::eCalcMethod& calcMethod, const eOrthogonalArray& orthArray = L_None);

	void SaveYoloExperiment_AnswerBox_txt(YOLO_v3_DNN* YOLOv3_Module_, const float& NMSThreshold, const float& confidenceThreshold);
	void SaveAnswer_AreaRatio_txt(const std::string& folder);
	void SaveColorDetection_DetectionBox_txt(ColorDetection* ColorDetection_Module,
		const int& hueMargin, const int& medianFilterSize, const int& downSamplingSize);
};