#pragma once

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "InBoxChecker_Tool.hpp"

class GrayScaleCalculator
{
public:
	enum eCalcMethod { RGB_Luminance = 0, HSV_Value, HSL_Lightness, CIELab_Lightness };
	struct Reference
	{
		eCalcMethod CalcMethod;
		float AverageBrightness;
	};
private:
	eCalcMethod mCurCalcMethod;

	float (GrayScaleCalculator::* mCalcfunctionPtr)(cv::Mat& currFrame, InBoxChecker* inBoxChecker);

	float calcByRGB_Luminance(cv::Mat& currFrame, InBoxChecker* inBoxChecker);
	float calcByHSV_Value(cv::Mat& currFrame, InBoxChecker* inBoxChecker);
	float calcByHSL_Lightness(cv::Mat& currFrame, InBoxChecker* inBoxChecker);
	float calcByCIELab_Lightness(cv::Mat& currFrame, InBoxChecker* inBoxChecker);

	std::vector < GrayScaleCalculator::Reference> mInnerReferences;

public:

	void SetCalcMethod(const eCalcMethod& calcMethod);
	eCalcMethod GetCalcMethod();

	GrayScaleCalculator(const eCalcMethod& calcMethod);
	float CalcGrayScale(cv::Mat& currFrame, InBoxChecker* inBoxChecker);

	void AddReference(const std::string& referenceAddress);
	void AddReference(const std::string& referenceAddress, const GrayScaleCalculator::eCalcMethod& calcMethod);
	std::vector<GrayScaleCalculator::Reference> GetReferences();
	GrayScaleCalculator::Reference GetReferenceAt(const int& index_);
	void ClearReferences();

};