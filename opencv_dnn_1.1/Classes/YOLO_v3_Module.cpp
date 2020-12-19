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


	void drawPred(const int& classId, const float& conf,const cv::Rect& box, cv::Mat* currFrame);
	std::vector<std::string> mGetOutputsNames(const cv::dnn::Net& net);

	void doPostprocessFrame(cv::Mat* currFrame);
	void doConfidenceProcess(cv::Mat* currFrame);
	void doNMSProcess();
	void drawFrameTime(cv::Mat* currFrame);
public:
	YOLO_v3_DNN(const float& confThreshold = 0.2,const float& nmsThreshold = 0.4,const int& inpWidth = 416,const int& inpHeight = 416);
	void MakeYOLONetFromFile(const std::string& classesFile, const std::string& modelConfiguration,const std::string& modelWeights);
	void PassThroughWithPostProcessing(const cv::Mat& currFrame, cv::Mat* detectedFrame);
	bool PassThrough(cv::Mat* currFrame);
	int GetObjectRects(InBoxChecker* inBoxChecker);
	void DrawBoxes(cv::Mat* currFrame, const bool& isDrawFrameTime = true);

	bool SetConfidenceThreshold(const float& confidenceThreshold);
	bool SetNMSThreshold(const float& NMSThreshold);
};

//생성자
YOLO_v3_DNN::YOLO_v3_DNN(const float& confThreshold,const float& nmsThreshold,const int& inpWidth,const int& inpHeight)
{
	this->mConfThreshold = confThreshold;
	this->mNMSThreshold = nmsThreshold;
	this->mInpWidth = inpWidth;
	this->mInpHeight = inpHeight;
}

//경로를 받아서 네트워크를 만든다.
void YOLO_v3_DNN::MakeYOLONetFromFile(const std::string& classesFile, const std::string& modelConfiguration,const std::string& modelWeights)
{
	//dnn network settings for object detection
	std::ifstream ifs(classesFile.c_str());
	std::string line;
	while (getline(ifs, line)) mClasses.push_back(line);

	mNeuralNet = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);

	if (mNeuralNet.empty())
	{
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "cfg-file:     " << modelConfiguration << std::endl;
		std::cerr << "weights-file: " << modelWeights << std::endl;
		std::cerr << "Models can be downloaded here:" << std::endl;
		std::cerr << "https://pjreddie.com/darknet/yolo/" << std::endl;
		exit(-1);
	}

	mNeuralNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	mNeuralNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

//Mat을 하나 받아서 네트워크에 통과시킨다음 박스를 그린다.
void YOLO_v3_DNN::PassThroughWithPostProcessing(const cv::Mat& currFrame, cv::Mat* detectedFrame)
{
	cv::Mat& detectedFrame_ = *detectedFrame;

	if (mNeuralNet.empty())
	{
		std::cerr << "Can't load the dnn net internally!" << std::endl;
		std::cerr << "make a proper net before use this function" << std::endl;
		std::cerr << "make a proper net with right filestring please" << std::endl;
	}
	else
	{
		//원본 프레임에서 detectedFrame으로 deep copy
		currFrame.copyTo(detectedFrame_);

		//do dnn calculation
		mBlob = cv::dnn::blobFromImage(currFrame, 1 / 255.0, cv::Size(mInpWidth, mInpHeight), cv::Scalar(0, 0, 0), true, false);
		mNeuralNet.setInput(mBlob);
		mNeuralNet.forward(mOuts, mGetOutputsNames(mNeuralNet));

		//draw boxes and labels
		doPostprocessFrame(&detectedFrame_);

		//draw frames
		drawFrameTime(&detectedFrame_);

		//frame_.convertTo(detectedFrame, CV_8U);

		//return detectedFrame_;

	}
}

//네트워크에 통과시키고 박스,인덱스들을 전부 만든다.
bool YOLO_v3_DNN::PassThrough(cv::Mat* currFrame)
{
	cv::Mat& currFrame_ = *currFrame;

	if (mNeuralNet.empty())
	{
		std::cerr << "Can't load the dnn net internally!" << std::endl;
		std::cerr << "make a proper net before use this function" << std::endl;
		std::cerr << "make a proper net with right filestring please" << std::endl;
		return false;
	}
	else
	{
		mBlob = cv::dnn::blobFromImage(currFrame_, 1 / 255.0, cv::Size(mInpWidth, mInpHeight), cv::Scalar(0, 0, 0), true, false);
		mNeuralNet.setInput(mBlob);
		mNeuralNet.forward(mOuts, mGetOutputsNames(mNeuralNet));

		doConfidenceProcess(&currFrame_);
		doNMSProcess();

		return true;
	}
}

//확률 그리기. postprocess함수에서 사용한다.
void YOLO_v3_DNN::drawPred(const int& classId, const float& conf, const cv::Rect& box, cv::Mat* currFrame)
{
	cv::Mat& currFrame_ = *currFrame;

	//Draw a rectangle displaying the bounding box
	cv::rectangle(currFrame_, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height),
		cv::Scalar(0, 0, 255));

	//Get the label for the class name and its confidence
	std::string label = cv::format("%.2f", conf);
	if (!mClasses.empty())
	{
		CV_Assert(classId < (int)mClasses.size());
		label = mClasses[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	int put_y = cv::max(box.y, labelSize.height);
	cv::putText(currFrame_, label, cv::Point(box.x, put_y), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 0, 0), 2);
}

//Net의 아웃풋 레이어 구조를 따오는 함수. Net에 이미지를 통과시킬때 사용한다.
std::vector<std::string> YOLO_v3_DNN::mGetOutputsNames(const cv::dnn::Net& net)
{
	static std::vector<std::string> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		std::vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		std::vector<std::string> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

//결과 값들의 박스를 치는 함수
void YOLO_v3_DNN::doPostprocessFrame(cv::Mat* currFrame)
{
	cv::Mat& currFrame_ = *currFrame;

	doConfidenceProcess(&currFrame_);
	doNMSProcess();
	DrawBoxes(&currFrame_);
}

//신뢰도를 기반으로 해서 박스들을 합쳐간다
void YOLO_v3_DNN::doConfidenceProcess(cv::Mat* currFrame)
{
	cv::Mat& currFrame_ = *currFrame;

	mClassIds.clear();
	mConfidences.clear();
	mBoxes.clear();

	for (size_t i = 0; i < mOuts.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)mOuts[i].data;
		cv::Mat scores;

		for (int j = 0; j < mOuts[i].rows; ++j, data += mOuts[i].cols)
		{
			scores = mOuts[i].row(j).colRange(5, mOuts[i].cols);
			cv::Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > mConfThreshold)
			{
				int centerX = (int)(data[0] * currFrame_.cols);
				int centerY = (int)(data[1] * currFrame_.rows);
				int width = (int)(data[2] * currFrame_.cols);
				int height = (int)(data[3] * currFrame_.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				mClassIds.push_back(classIdPoint.x);
				mConfidences.push_back((float)confidence);
				mBoxes.push_back(cv::Rect(left, top, width, height));
			}
		}
	}
}

//계산된 값을 바탕으로 NMS 알고리즘을 거친다.
//indices에 적혀있는 인덱스가 진짜 Rect로써 바뀐다.
void YOLO_v3_DNN::doNMSProcess()
{
	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	mIndices.clear();
	cv::dnn::NMSBoxes(mBoxes, mConfidences, mConfThreshold, mNMSThreshold, mIndices);

}

//내부에 저장된 Rect값들을 뽑아온다
int YOLO_v3_DNN::GetObjectRects(InBoxChecker* inBoxChecker)
{
	InBoxChecker& inBoxChecker_ = *inBoxChecker;

	int classId;
	std::string label;
	for (size_t i = 0; i < mIndices.size(); ++i)
	{
		classId = mClassIds[mIndices[i]];
		CV_Assert(classId < (int)mClasses.size());
		label = mClasses[classId];

		inBoxChecker_.AddBox(mBoxes[mIndices[i]], InBoxChecker::YOLOv3, label);
	}
}

//내부에 저장된 Rect값들을 이용해서 박스를 그린다.
void YOLO_v3_DNN::DrawBoxes(cv::Mat* currFrame,const bool& isDrawFrameTime) 
{
	for (size_t i = 0; i < mIndices.size(); ++i)
	{
		int idx = mIndices[i];
		cv::Rect box = mBoxes[idx];
		drawPred(mClassIds[idx], mConfidences[idx], box, currFrame);
	}

	if (isDrawFrameTime)
	{
		drawFrameTime(currFrame);
	}
}

//사진 맨 위에 프레임타임을 쓴다
void YOLO_v3_DNN::drawFrameTime(cv::Mat* currFrame)
{
	std::vector<double> layersTimes;
	double freq = cv::getTickFrequency() / 1000;
	double t = mNeuralNet.getPerfProfile(layersTimes) / freq;
	std::string label = cv::format("Inference time for a frame : %.2f ms", t);
	cv::putText(*currFrame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
}

bool YOLO_v3_DNN::SetConfidenceThreshold(const float& confidenceThreshold)
{
	if ((0.0f <= confidenceThreshold) && (confidenceThreshold <= 1.0f))
	{
		mConfThreshold = confidenceThreshold;
		return true;
	}
	else
	{
		return false;
	}
}

bool YOLO_v3_DNN::SetNMSThreshold(const float& NMSThreshold)
{
	if ((0.0f <= NMSThreshold) && (NMSThreshold <= 1.0f))
	{
		mNMSThreshold = NMSThreshold;
		return true;
	}
	else
	{
		return false;
	}
}