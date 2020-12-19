
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

//생성자
GrayScaleCalculator::GrayScaleCalculator(const eCalcMethod& calcMethod)
{
	SetCalcMethod(calcMethod);
};

//계산 방법을 세팅하는 함수
void GrayScaleCalculator::SetCalcMethod(const eCalcMethod& calcMethod)
{
	switch (calcMethod)
	{
	case RGB_Luminance:
	{
		mCalcfunctionPtr = &GrayScaleCalculator::calcByRGB_Luminance;
		break;
	}
	case HSV_Value:
	{
		mCalcfunctionPtr = &GrayScaleCalculator::calcByHSV_Value;
		break;
	}
	case HSL_Lightness:
	{
		mCalcfunctionPtr = &GrayScaleCalculator::calcByHSL_Lightness;
		break;
	}
	case CIELab_Lightness:
	{
		mCalcfunctionPtr = &GrayScaleCalculator::calcByCIELab_Lightness;
		break;
	}
	}

	mCurCalcMethod = calcMethod;
};

//계산 방법을 반환하는 함수
GrayScaleCalculator::eCalcMethod GrayScaleCalculator::GetCalcMethod()
{
	return mCurCalcMethod;
};

//그레이스케일을 계산하는 함수
float GrayScaleCalculator::CalcGrayScale(cv::Mat& currFrame, InBoxChecker* inBoxChecker)
{
	auto& inBoxChecker_ = *inBoxChecker;
	float grayscaleValue;
	grayscaleValue = (this->*mCalcfunctionPtr)(currFrame, &inBoxChecker_);
	return grayscaleValue;
};

//그레이스케일을 RGB_Luminance로 계산하는 함수
float GrayScaleCalculator::calcByRGB_Luminance(cv::Mat& currFrame, InBoxChecker* inBoxChecker)
{
	auto& inBoxChecker_ = *inBoxChecker;

	cv::Vec3b* BGRPtr;
	double result = 0.0;
	int count = 1;
	float luminance = 0;
	int boxwidth = 0;
	for (int r = 0; r < currFrame.rows; r++)
	{
		for (int c = 0; c < currFrame.cols; c++)
		{
			if (inBoxChecker_.CheckInBox(cv::Point(c, r), &boxwidth))
			{
				c += boxwidth;
				continue;
			}
			BGRPtr = currFrame.ptr<cv::Vec3b>(r);
			luminance = 0.2126f * (int)(BGRPtr[c][2]) + 0.7152f * (int)(BGRPtr[c][1]) + 0.0722f * (int)(BGRPtr[c][0]);
			result = (result / count * (count - 1)) + (luminance / count);
			count++;
		}
	}

	return result;
};

float GrayScaleCalculator::calcByHSV_Value(cv::Mat& currFrame, InBoxChecker* inBoxChecker)
{
	auto& inBoxChecker_ = *inBoxChecker;

	cv::Mat HSVFrame;
	cv::cvtColor(currFrame, HSVFrame, cv::COLOR_BGR2HSV);
	cv::Vec3b* HSVPtr;
	double result = 0.0;
	int count = 1;
	int boxwidth;
	for (int r = 0; r < currFrame.rows; r++)
	{
		for (int c = 0; c < currFrame.cols; c++)
		{
			if (inBoxChecker_.CheckInBox(cv::Point(c, r), &boxwidth))
			{
				c += boxwidth;
				continue;
			}
			HSVPtr = HSVFrame.ptr<cv::Vec3b>(r);
			result = (result * (count - 1) / (float)count) + ((float)HSVPtr[c][2] / count);
			count++;
		}
	}

	return result;
}

float GrayScaleCalculator::calcByHSL_Lightness(cv::Mat& currFrame, InBoxChecker* inBoxChecker)
{
	auto& inBoxChecker_ = *inBoxChecker;

	cv::Mat HLSFrame;
	cv::cvtColor(currFrame, HLSFrame, cv::COLOR_BGR2HLS);
	cv::Vec3b* HLSPtr;
	double result = 0.0;
	int count = 1;
	int boxwidth;
	for (int r = 0; r < currFrame.rows; r++)
	{
		for (int c = 0; c < currFrame.cols; c++)
		{
			if (inBoxChecker_.CheckInBox(cv::Point(c, r), &boxwidth))
			{
				c += boxwidth;
				continue;
			}
			HLSPtr = HLSFrame.ptr<cv::Vec3b>(r);
			result = (result * (count - 1) / (float)count) + ((float)HLSPtr[c][1] / count);
			count++;
		}
	}

	return result;
}

float GrayScaleCalculator::calcByCIELab_Lightness(cv::Mat& currFrame, InBoxChecker* inBoxChecker)
{
	auto& inBoxChecker_ = *inBoxChecker;

	cv::Mat LabFrame;
	cv::cvtColor(currFrame, LabFrame, cv::COLOR_BGR2Lab);
	cv::Vec3b* LabPtr;
	double result = 0.0;
	int count = 1;
	int boxwidth;
	for (int r = 0; r < currFrame.rows; r++)
	{
		for (int c = 0; c < currFrame.cols; c++)
		{
			if (inBoxChecker_.CheckInBox(cv::Point(c, r), &boxwidth))
			{
				c += boxwidth;
				continue;
			}
			LabPtr = LabFrame.ptr<cv::Vec3b>(r);
			result = (result * (count - 1) / (float)count) + ((float)LabPtr[c][0] / count);
			count++;
		}
	}

	return result;
}

//fGray = 0.2126f * chRed + 0.7152f * chGreen + 0.0722f * chBlue;

//현재의 칼큘레이션 방법을 유지하면서 진행
void GrayScaleCalculator::AddReference(const std::string& referenceAddress)
{
	InBoxChecker inboxChecker;
	inboxChecker.ClearBox();

	cv::Mat referenceImg = cv::imread(referenceAddress);
	float reference_value = CalcGrayScale(referenceImg, &inboxChecker);
	
	Reference newRef;
	newRef.AverageBrightness = reference_value;
	newRef.CalcMethod = mCurCalcMethod;
	mInnerReferences.push_back(newRef);
}

//새롭게 칼큘레이션 방법을 바꾸면서 진행
void GrayScaleCalculator::AddReference(const std::string& referenceAddress, const GrayScaleCalculator::eCalcMethod& calcMethod)
{
	InBoxChecker inboxChecker;
	inboxChecker.ClearBox();

	cv::Mat referenceImg = cv::imread(referenceAddress);
	SetCalcMethod(calcMethod);
	float reference_value = CalcGrayScale(referenceImg, &inboxChecker);

	Reference new_ref;
	new_ref.AverageBrightness = reference_value;
	new_ref.CalcMethod = mCurCalcMethod;
	mInnerReferences.push_back(new_ref);
}

std::vector<GrayScaleCalculator::Reference> GrayScaleCalculator::GetReferences()
{
	return mInnerReferences;
}

GrayScaleCalculator::Reference GrayScaleCalculator::GetReferenceAt(const int& index)
{
	return mInnerReferences[index];
}

void GrayScaleCalculator::ClearReferences()
{
	mInnerReferences.clear();
}


