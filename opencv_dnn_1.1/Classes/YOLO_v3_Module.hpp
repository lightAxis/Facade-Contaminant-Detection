#pragma once

#include <opencv2\dnn.hpp>
#include <opencv2\dnn/shape_utils.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <algorithm> 
#include <queue>
#include <sstream>
#include <iostream>
#include <fstream>

#include "InBoxChecker_Tool.hpp"


class YOLO_v3_DNN
{
private:
	float mConfThreshold = 0.2; // Confidence threshold
	float mNMSThreshold = 0.4;  // Non-maximum suppression threshold
	int mInpWidth = 416;        // Width of network's input image
	int mInpHeight = 416;       // Height of network's input image

	cv::dnn::Net mNeuralNet;

	cv::Mat mBlob;
	std::vector<cv::Mat> mOuts;
	cv::Mat mDetectedFrame;

	std::vector<int> mClassIds;
	std::vector<float> mConfidences;
	std::vector<cv::Rect> mBoxes;
	std::vector<std::string> mClasses;
	std::vector<int> mIndices;


	void drawPred(const int& classId, const float& conf, const cv::Rect& box, cv::Mat* currFrame);
	std::vector<std::string> mGetOutputsNames(const cv::dnn::Net& net);

	void doPostprocessFrame(cv::Mat* currFrame);
	void doConfidenceProcess(cv::Mat* currFrame);
	void doNMSProcess();
	void drawFrameTime(cv::Mat* currFrame);
public:
	YOLO_v3_DNN(const float& confThreshold = 0.2, const float& nmsThreshold = 0.4, const int& inpWidth = 416, const int& inpHeight = 416);
	void MakeYOLONetFromFile(const std::string& classesFile, const std::string& modelConfiguration, const std::string& modelWeights);
	void PassThroughWithPostProcessing(const cv::Mat& currFrame, cv::Mat* detectedFrame);
	bool PassThrough(cv::Mat* currFrame);
	int GetObjectRects(InBoxChecker* inBoxChecker);
	void DrawBoxes(cv::Mat* currFrame, const bool& isDrawFrameTime = true);

	bool SetConfidenceThreshold(const float& confidenceThreshold);
	bool SetNMSThreshold(const float& NMSThreshold);
};