#pragma once

#include <vector>
#include <queue>
#include <io.h>	
#include <tuple>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>


#include "InBoxChecker_Tool.hpp"


class ColorDetection
{
private:
	struct meHueDetectionRange
	{
		int mPosZeroPoint;
		int mNegZeroPoint;
		int mMidOnePoint;

		bool mbIncludeZero;
		bool mbMidBeforeZero;

		float mAPos;
		float mXPosBias;

		float mANeg;
		float mXNegBias;

		int mSaturationThreshold;

		std::string mName;
	};

	std::vector<meHueDetectionRange> mHueDetectionRanges;
	float colorHueMapping(float mappingValue, const int& curSaturation, meHueDetectionRange* currHueDetectionRange);

	int mMedianBlurSize;
	int mDownSamplingSize;
	int mDownSamplingSize_Half;
	std::vector<uint8_t> mMedianBlurTempVector;

	std::vector<cv::Mat> mBackgroundSamples;
	std::vector<cv::Mat> mColorSamples;

	std::vector<std::string> getImageAddressFromFolder(
		const std::string& FolderAddress,
		std::vector<std::vector<std::string>>* backgroudAddresses,
		std::vector<std::vector<std::string>>* ColorAreaAddresses);

	std::vector<std::tuple<int, int, int>> getThresholds(
		std::vector<std::vector<std::string>>* backAddresses,
		std::vector<std::vector<std::string>>* colorAddresses,
		std::vector<std::tuple<int, int>>* backThresholds,
		std::vector<std::tuple<int, int>>* colorThresholds);

public:
	enum eMedianBlurSizes {
		F3x3 = 3, F5x5 = 5, F7x7 = 7, F9x9 = 9, F11x11 = 11, F13x13 = 13, F15x15 = 15, F17x17 = 17,
		F19x19 = 19, F21x21 = 21, F23x23 = 23, F25x25 = 25, F27x27 = 27, F29x29 = 29, F31x31 = 31, F33x33 = 33
	};
	enum eDownSamplingSizes {
		S3x3 = 3, S5x5 = 5, S7x7 = 7, S9x9 = 9, S11x11 = 11, S13x13 = 13, S15x15 = 15, S17x17 = 17,
		S19x19 = 19, S21x21 = 21, S23x23 = 23, S25x25 = 25, S27x27 = 27, S29x29 = 29, S31x31 = 31, S33x33 = 33
	};

	ColorDetection(const eMedianBlurSizes& MedianBlurSize_ = F9x9, const eDownSamplingSizes& SamplingSize_ = S11x11);
	ColorDetection(const int& positivePointZero, const int& midPointOne, const int& negativePointZero,
		const std::string& name, const eMedianBlurSizes& medianBlurSize = F9x9, const eDownSamplingSizes& samplingSize = S11x11);

	void AddColorDetectionRange(const int& positivePointZero, const int& midPointOne, const int& negativePointZero,
		const std::string& name, const int& saturationThreshold = 40);
	void ClearColorDetectionRange();
	void SetMedianBlurSize(const eMedianBlurSizes& medianBlurSize);
	void SetDownSamplingSize(const eDownSamplingSizes& samplingSize);
	int PushThroughImage(const cv::Mat& testFrame, cv::Mat* HSVedFrame,
		InBoxChecker* inBoxChecker, const int& detectionIndex = 0);

	int MakeBoxWithMedianBlur(cv::Mat* hueDetectedFrame, cv::Mat* boxFlagFrame,
		InBoxChecker* inBoxChecker, const std::string& name);
	uint8_t DoMedianBlur(cv::Mat* frame, const cv::Point& pt, const int& medianBlurHalfSize);
	int MakeBoxWithFloodFill(cv::Mat* boxFlagFrame, InBoxChecker* inBoxChecker,
		const int& originalBoxRows, const int& originalBoxCols, const std::string& name);

	void MakeColorDetection(const std::string& colorSampleFolder, const int& hueMargin = 8);
};