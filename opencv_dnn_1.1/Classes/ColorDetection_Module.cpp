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
	float colorHueMapping(float mappingValue,const int& curSaturation, meHueDetectionRange* currHueDetectionRange);

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
	enum eMedianBlurSizes { F3x3 = 3, F5x5 = 5, F7x7 = 7, F9x9 = 9, F11x11 = 11, F13x13 = 13, F15x15 = 15, F17x17 = 17, 
		F19x19 = 19, F21x21 = 21, F23x23 = 23, F25x25 = 25, F27x27 = 27, F29x29 = 29, F31x31 = 31, F33x33 = 33};
	enum eDownSamplingSizes { S3x3 = 3, S5x5 = 5, S7x7 = 7, S9x9 = 9, S11x11 = 11, S13x13 = 13, S15x15 = 15, S17x17 = 17,
		S19x19 = 19, S21x21 = 21, S23x23 = 23, S25x25 = 25, S27x27 = 27, S29x29 = 29, S31x31 = 31, S33x33 = 33};

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

//생성자
ColorDetection::ColorDetection(const eMedianBlurSizes& medianBlurSize, const eDownSamplingSizes& samplingSize)
{
	mHueDetectionRanges.clear();
	mMedianBlurSize = medianBlurSize;
	SetDownSamplingSize(samplingSize);
	mMedianBlurTempVector = std::vector<uint8_t>(medianBlurSize * medianBlurSize);
}

ColorDetection::ColorDetection(const int& positivePointZero, const int& midPointOne, const int& negativePointZero,
	const std::string& name, const eMedianBlurSizes& medianBlurSize, const eDownSamplingSizes& samplingSize)
{
	mHueDetectionRanges.clear();
	AddColorDetectionRange(positivePointZero, midPointOne, negativePointZero, name);
	mMedianBlurSize = medianBlurSize;
	SetDownSamplingSize(samplingSize);
	mMedianBlurTempVector = std::vector<uint8_t>(medianBlurSize * medianBlurSize);
}


//새로운 컬러 디텍션 범위를 만든다.
void ColorDetection::AddColorDetectionRange(const int& positivePointZero, const int& midPointOne, const int& negativePointZero,
	const std::string& name, const int& saturationThreshold)
{
	meHueDetectionRange newRange = meHueDetectionRange();
	newRange.mPosZeroPoint = positivePointZero;
	newRange.mMidOnePoint = midPointOne;
	newRange.mNegZeroPoint = negativePointZero;
	newRange.mSaturationThreshold = saturationThreshold;

	if (positivePointZero < negativePointZero)
	{
		newRange.mbIncludeZero = true;
		newRange.mNegZeroPoint -= 255;
		if (positivePointZero < midPointOne)
		{
			newRange.mbMidBeforeZero = true;
			newRange.mMidOnePoint -= 255;
		}
	}

	newRange.mAPos = (float)(-1.0 / (newRange.mPosZeroPoint - newRange.mMidOnePoint));
	newRange.mXPosBias = (float)newRange.mPosZeroPoint;

	newRange.mANeg = (float)(1.0 / (newRange.mMidOnePoint - newRange.mNegZeroPoint));
	newRange.mXNegBias = (float)newRange.mNegZeroPoint;

	newRange.mName = name;


	mHueDetectionRanges.push_back(newRange);
}

//컬러 디텍션 범위를 이용해서 HUE값을 매핑하는 함수
float ColorDetection::colorHueMapping(float mappingValue, const int& curSaturation, meHueDetectionRange* curHueDetectionRange)
{
	meHueDetectionRange& curHueDetectionRange_ = *curHueDetectionRange;

	if (mappingValue > curHueDetectionRange_.mPosZeroPoint)
	{
		mappingValue -= 255.0;
	}

	//범위 안에 있는지 검사
	if ((curHueDetectionRange_.mNegZeroPoint < mappingValue) && (mappingValue < curHueDetectionRange_.mPosZeroPoint))
	{
		//새츄레이션 쓰레시홀드를 넘었는지 검사
		if (curSaturation >= curHueDetectionRange_.mSaturationThreshold)
		{
			//pos방향 매핑
			if (curHueDetectionRange_.mMidOnePoint < mappingValue)
			{
				return curHueDetectionRange_.mAPos* (mappingValue - curHueDetectionRange_.mXPosBias);

			}
			else //neg방향 매핑
			{
				return curHueDetectionRange_.mANeg* (mappingValue - curHueDetectionRange_.mXNegBias);
			}
		}
	}
	return 0.0;
}

//컬러 디텍션 레인지를 초기화
void ColorDetection::ClearColorDetectionRange()
{
	mHueDetectionRanges.clear();
}

//메디안 블러 필터 사이즈 설정
void ColorDetection::SetMedianBlurSize(const eMedianBlurSizes& medianBlurSize)
{
	mMedianBlurSize = medianBlurSize;
	mMedianBlurTempVector = std::vector<uint8_t>((int)medianBlurSize * medianBlurSize);
}

void ColorDetection::SetDownSamplingSize(const eDownSamplingSizes& samplingSize)
{
	mDownSamplingSize = samplingSize;
	mDownSamplingSize_Half = samplingSize / 2;
}

//이미지를 프로세스 하는 함수
int ColorDetection::PushThroughImage(const cv::Mat& testFrame, cv::Mat* HSVedFrame, 
	InBoxChecker* inBoxChecker, const int& detectionIndex)
{
	auto& HSVedFrame_ = *HSVedFrame;
	auto& inBoxChecker_ = *inBoxChecker;

	cv::Mat HSV_Frame = cv::Mat(testFrame.rows, testFrame.cols, CV_8UC3);
	cv::Mat detectedFrame = cv::Mat::zeros(testFrame.rows, testFrame.cols, CV_8UC1);
	//cv::Mat medianBluredFrame;
	cv::cvtColor(testFrame, HSV_Frame, cv::COLOR_BGR2HSV);
	cv::Vec3b* HSVframePtr;
	uchar* detectedFramePtr = detectedFrame.ptr(0);

	float hueVal;
	//int jumping_width;

	int flagRow = (testFrame.rows - mDownSamplingSize_Half - 1) / mDownSamplingSize + 1;
	int flagCol = (testFrame.cols - mDownSamplingSize_Half - 1) / mDownSamplingSize + 1;
	cv::Mat BoxFlagFrame = cv::Mat::zeros(flagRow + 2, flagCol + 2, CV_8UC1);

	//Hue컬러값 지도를 만드는 과정
	for (int r = 0; r < HSV_Frame.rows; r++)
	{
		HSVframePtr = HSV_Frame.ptr<cv::Vec3b>(r);
		detectedFramePtr = detectedFrame.ptr(r);
		for (int c = 0; c < HSV_Frame.cols; c++)
		{

			hueVal = colorHueMapping(HSVframePtr[c][0], HSVframePtr[c][1], &mHueDetectionRanges[detectionIndex]);
			//127까지만 하는 이유는 맨 앞비트를 Flag로 쓰기 위해서이다.
			detectedFramePtr[c] = (int)(hueVal * 127);
		}
	}


	//메디안 블러된 점을 다운샘플링 간격 뽑아내어 따로 저장한다.
	//반환값은 플래그맵에서 감지된 총 컬러영역 픽셀의 개수*다운샘플링 크기^2 으로 계산된다.
	int detectedPixelCount = MakeBoxWithMedianBlur(&detectedFrame, &BoxFlagFrame, &inBoxChecker_ ,mHueDetectionRanges[detectionIndex].mName);
	cv::Mat testTempFrame;
	cv::resize(BoxFlagFrame, testTempFrame, cv::Size(detectedFrame.cols, detectedFrame.rows), 0, 0, cv::InterpolationFlags::INTER_LINEAR);
	cv::imshow("sampling", testTempFrame);
	cv::imwrite("./imwrite_imgs/sampling.jpg", testTempFrame);

	cv::imshow("HSV_Hue mapped frame", detectedFrame);
	cv::imwrite("./imwrite_imgs/HSV_Hue mapped frame.jpg", detectedFrame);

	cv::Mat medianBluredFrame;
	cv::medianBlur(detectedFrame, medianBluredFrame, mMedianBlurSize);
	cv::imshow("sampling_medianBlured", medianBluredFrame);
	cv::imwrite("./imwrite_imgs/sampling_medianBlured.jpg", medianBluredFrame);


	HSV_Frame.copyTo(HSVedFrame_);
	return detectedPixelCount;
};

//메디안 블러를 사용해서 박스를 만든다. 내부적으로는 FloodFill 알고리즘도 사용되었다.
int ColorDetection::MakeBoxWithMedianBlur(cv::Mat* hueDetectedFrame, cv::Mat* boxFlagFrame, 
	InBoxChecker* inBoxChecker,const std::string& name)
{
	auto& hueDetectedFrame_ = *hueDetectedFrame;
	auto& boxFlagFrame_ = *boxFlagFrame;
	auto& inBoxChecker_ = *inBoxChecker;


	int jumpingWidth = 0;
	int medianBlurHalf = mMedianBlurSize / 2;

	uchar* boxFlagFramePtr;
	uchar* detectedFramePtr;

	int flagRow = 1;
	int flagCol = 1;

	for (int r = mDownSamplingSize_Half; r < hueDetectedFrame_.rows; r += mDownSamplingSize)
	{
		detectedFramePtr = hueDetectedFrame_.ptr(r);
		boxFlagFramePtr = boxFlagFrame_.ptr(flagRow);

		flagCol = 1;
		for (int c = mDownSamplingSize_Half; c < hueDetectedFrame_.cols; c += mDownSamplingSize)
		{
			if (inBoxChecker_.CheckInBox(cv::Point(c, r), &jumpingWidth))
			{
				c = c + ((((jumpingWidth) / mDownSamplingSize) + 1) * mDownSamplingSize);
				flagCol = ((c - 4) / mDownSamplingSize) + 1;
				flagCol++;
				continue;
			}
			boxFlagFramePtr[flagCol] = DoMedianBlur(&hueDetectedFrame_, cv::Point(c, r), medianBlurHalf);
			auto asdf = boxFlagFramePtr[flagCol];

			flagCol++;
		}
		flagRow++;
	}

	int totalDetectedPixelCount = MakeBoxWithFloodFill(&boxFlagFrame_, &inBoxChecker_, hueDetectedFrame_.rows, hueDetectedFrame_.cols,name);

	return totalDetectedPixelCount;
};

//한번 값이 다 들어간 박스플래그매트릭스를 이용해서 박스를 만들어서 인박스체커에 등록하는 함수
//반환값은 감지된 총 픽셀 수이다.
int ColorDetection::MakeBoxWithFloodFill(cv::Mat* boxFlagFrame, InBoxChecker* inBoxChecker, 
	const int& originalBoxRows, const int& originalBoxCols, const std::string& name)
{
	auto& boxFlagFrame_ = *boxFlagFrame;
	auto& inBoxChecker_ = *inBoxChecker;

	uchar* boxFlagFramePtr;

	std::queue<cv::Point> pointQue;
	uchar val;
	int addWidth;

	cv::Point curPt;

	cv::Point pt_Up;
	cv::Point pt_Down;
	cv::Point pt_Left;
	cv::Point pt_Right;

	cv::Point pt_UpLeft;
	cv::Point pt_DownRight;

	int totalDetectedPixel = 0;

	for (int r = 1; r < boxFlagFrame_.rows - 1; r++)
	{
		boxFlagFramePtr = boxFlagFrame_.ptr(r);
		for (int c = 1; c < boxFlagFrame_.cols - 1; c++)
		{
			//이미 있는 박스에 포함된다면 빠르게 넘긴다
			if (inBoxChecker_.CheckInBox(cv::Point((c - 1) * mDownSamplingSize + mDownSamplingSize_Half,
				(r - 1) * mDownSamplingSize + mDownSamplingSize_Half), &addWidth))
			{
				c = c + ((addWidth / mDownSamplingSize) + 1);
				continue;
			}

			val = boxFlagFramePtr[c];

			//floodfill 시작
			if (val > 0)
			{
				//차피 안비어 있다면 이 앞의 루프를 영원히 돌았을 것이므로.. 굳이 새롭게 초기화할 필요는 없겠다.
				pointQue.push(cv::Point(c, r));

				pt_UpLeft = cv::Point(c, r);
				pt_DownRight = cv::Point(c, r);

				boxFlagFrame_.at<uchar>(cv::Point(c, r)) += 0x80;

				while (pointQue.empty() == false)
				{
					//일단 픽셀이 감지되었으니 하나 추가하고 시작
					totalDetectedPixel++;

					curPt = pointQue.front();
					pointQue.pop();

					pt_Up = curPt;
					pt_Up.y -= 1;

					pt_Down = curPt;
					pt_Down.y += 1;

					pt_Left = curPt;
					pt_Left.x -= 1;

					pt_Right = curPt;
					pt_Right.x += 1;

					//오른쪽 검사,맨 위의 비트인 128을 의미하는 비트가 존재하는지 검사한다. 이것이 자체 flag로 동작한다.
					if (boxFlagFrame_.at<uchar>(pt_Right) > 0)
					{
						if ((boxFlagFrame_.at<uchar>(pt_Right) >> 7) == 0x00)
						{
							//검사가 안되어있었다면, 큐에 새롭게 넣고,
							pointQue.push(pt_Right);
							if (pt_Right.x > pt_DownRight.x) pt_DownRight.x = pt_Right.x;
							boxFlagFrame_.at<uchar>(pt_Right) += 0x80;
						}
					}
					//아래쪽 검사
					if (boxFlagFrame_.at<uchar>(pt_Down) > 0)
					{
						if ((boxFlagFrame_.at<uchar>(pt_Down) >> 7) == 0x00)
						{
							pointQue.push(pt_Down);
							if (pt_Down.y > pt_DownRight.y) pt_DownRight.y = pt_Down.y;
							boxFlagFrame_.at<uchar>(pt_Down) += 0x80;
						}
					}
					//왼쪽 검사
					if (boxFlagFrame_.at<uchar>(pt_Left) > 0)
					{
						if ((boxFlagFrame_.at<uchar>(pt_Left) >> 7) == 0x00)
						{
							pointQue.push(pt_Left);
							if (pt_Left.x < pt_UpLeft.x) pt_UpLeft.x = pt_Left.x;
							boxFlagFrame_.at<uchar>(pt_Left) += 0x80;
						}
					}
					//위쪽 검사
					if (boxFlagFrame_.at<uchar>(pt_Up) > 0)
					{
						if ((boxFlagFrame_.at<uchar>(pt_Up) >> 7) == 0x00)
						{
							pointQue.push(pt_Up);
							if (pt_Up.y < pt_UpLeft.y) pt_UpLeft.y = pt_Up.y;
							boxFlagFrame_.at<uchar>(pt_Up) += 0x80;
						}
					}
				}

				//박스들의 원래 크기+겉에 한칸씩 다 들리기
				pt_UpLeft = (pt_UpLeft - cv::Point(2, 2)) * mDownSamplingSize + cv::Point(mDownSamplingSize_Half, mDownSamplingSize_Half);
				pt_DownRight = (pt_DownRight * mDownSamplingSize + cv::Point(mDownSamplingSize_Half, mDownSamplingSize_Half));

				//원래 기존의 범위를 벗어나진 않았는지 검사
				if (pt_UpLeft.x < 0) pt_UpLeft.x = 0;
				if (pt_UpLeft.y < 0) pt_UpLeft.y = 0;
				if (pt_DownRight.x > originalBoxCols - 1) pt_DownRight.x = originalBoxCols - 1;
				if (pt_DownRight.y > originalBoxRows - 1) pt_DownRight.y = originalBoxRows - 1;



				cv::Rect newRect = cv::Rect(pt_UpLeft, pt_DownRight);
				inBoxChecker_.AddBox(newRect, InBoxChecker::ColorDetection, name);


			}

		}
	}

	//한번에 감지된 픽셀의 개수가 원래 몇개였는지 알 방법이 없다.
	//따라서 플래그맵의 다운샘플링 사이즈를 역이용하여, 플래그맵의 1칸당 다운샘플링사이즈^2개의 픽셀을 감지한것으로 간주한다.
	//속도는 빨라지겠지만, 대신 정확도는 줄어드는 트레이드 오프가 생긴다.
	totalDetectedPixel = totalDetectedPixel * mDownSamplingSize * mDownSamplingSize;

	return totalDetectedPixel;
};

//특정 점에서 메디안 블러를 수행하는 함수. 마이너스 좌표에 대해서는 답이없다.
uint8_t ColorDetection::DoMedianBlur(cv::Mat* frame, const cv::Point& pt, const int& medianBlurHalfSize)
{
	auto& frame_ = *frame;

	uchar* framePtr;;// = frame.ptr<uchar>(0);
	uint8_t i = 0;
	uint8_t correct = 0;

	for (int r = pt.y - medianBlurHalfSize; r <= pt.y + medianBlurHalfSize; r++)
	{

		for (int c = pt.x - medianBlurHalfSize; c <= pt.x + medianBlurHalfSize; c++)
		{
			if ((r >= frame_.rows) || (c >= frame_.cols) || (r < 0) || (c < 0))
			{
				mMedianBlurTempVector[i] = 0xFF;
				i++;
				continue;
			}
			framePtr = frame_.ptr<uchar>(r);
			mMedianBlurTempVector[i] = framePtr[c];
			i++;
			correct++;
		}
	}

	std::sort(mMedianBlurTempVector.begin(), mMedianBlurTempVector.end());

	return mMedianBlurTempVector[(correct + 1) / 2 - 1];

};


//컬러 디텍션 파라미터를 자동으로 결정해주는 함수
//폴더 상대경로 이름, 내부엔 원하는 클래스 숫자만큼 폴더 구현,그 안에는 다시 Background와 ColorArea폴더 구현해놓을것
//이미지 파일은 .png .bmp. jpg만 읽어들임
void ColorDetection::MakeColorDetection(const std::string& colorSampleFolder, const int& hueMargin)
{
	std::vector<std::vector<std::string>> backgroundImageAddresses;
	std::vector < std::vector<std::string>> colorSampleImageAddresses;
	std::vector<std::string> classes;

	classes = getImageAddressFromFolder(colorSampleFolder, &backgroundImageAddresses, &colorSampleImageAddresses);

	//<젤 높은 sat, 95%로 낮은 sat
	std::vector<std::tuple<int, int>> backgroundSatThresholds;
	//젤 낮은 sat, 95%로 높은 sat
	std::vector<std::tuple<int, int>> colorAreaSatThresholds;
	//positive, mid ,negative hue 범위임
	std::vector<std::tuple<int, int, int>> colorAreaHueMappings;

	colorAreaHueMappings = getThresholds(&backgroundImageAddresses, &colorSampleImageAddresses, &backgroundSatThresholds, &colorAreaSatThresholds);

	bool isHueAlreadyExist = false;

	for (int i = 0; i < classes.size(); i++)
	{
		isHueAlreadyExist = false;
		for (int ii = 0; ii < mHueDetectionRanges.size(); ii++)
		{
			if (mHueDetectionRanges[ii].mName == classes[i])
			{
				isHueAlreadyExist = true;
				break;
			}
		}

		if (isHueAlreadyExist) continue;

		uint8_t satThreshold = 0;
		uint8_t back_top = std::get<0>(backgroundSatThresholds[i]);
		uint8_t back_bot_confidence = std::get<1>(backgroundSatThresholds[i]);
		uint8_t color_bot = std::get<0>(colorAreaSatThresholds[i]);
		uint8_t color_top_confidence = std::get<1>(colorAreaSatThresholds[i]);

		if (back_top <= color_bot)
		{
			satThreshold = color_bot;
		}
		else if (color_top_confidence <= back_bot_confidence)
		{
			satThreshold = color_top_confidence;
		}
		else if (back_bot_confidence <= color_bot)
		{
			if (back_top <= color_top_confidence)
			{
				satThreshold = back_top;
			}
			else
			{
				satThreshold = color_top_confidence;
			}
		}
		else if (color_bot < back_bot_confidence)
		{
			satThreshold = back_bot_confidence;
		}


		uint8_t hue_pos = std::get<0>(colorAreaHueMappings[i]) + hueMargin;
		if (hue_pos > 0xFF) hue_pos -= 0xFF;
		uint8_t hue_neg = std::get<2>(colorAreaHueMappings[i]) - hueMargin;
		if (hue_neg < 0) hue_neg += 0xFF;


		AddColorDetectionRange(hue_pos, std::get<1>(colorAreaHueMappings[i]),
			hue_neg, classes[i], satThreshold);
	}
}

//백그라운드 이미지파일 경로들, 컬러영역 이미지파일 경로들을 ref로 가져오고, classes를 반환하는 함수
std::vector<std::string> ColorDetection::getImageAddressFromFolder(const std::string& folderAddress, 
	std::vector<std::vector<std::string>>* backgroudAddresses, 
	std::vector<std::vector<std::string>>* colorAreaAddresses)
{
	auto& backgroudAddresses_ = *backgroudAddresses;
	auto& colorAreaAddresses_ = *colorAreaAddresses;

	std::string searching = ".\\" + "autoColorParameters" +"\\" + folderAddress + "\\" + "*.*";
	std::vector<std::string> colorFolders;
	std::vector<std::string> classes;

	_finddata_t fd;
	intptr_t handle;
	int result;
	handle = _findfirst(searching.c_str(), &fd);

	std::string fileName;
	const char* tempChar;

	//내부에 원하는 클래스 폴더 탐지
	if (handle == -1)
	{
		printf("there is no File!\n");
	}

	while (true)
	{
		fileName = fd.name;
		std::cout << fileName << std::endl;
		tempChar = strrchr(fileName.c_str(), '.');

		if (tempChar == NULL)
		{
			handle = _findfirst((".\\" + folderAddress + "\\" + fileName + "*.*").c_str(), &fd);
			if (handle == -1)
			{
				std::cout << "this is not folder! : " << fileName << std::endl;
			}
			else
			{
				colorFolders.push_back(".\\" + folderAddress + "\\" + fileName);
				classes.push_back(fileName);
			}
		}

		result = _findnext(handle, &fd);
		if (result != 0)
		{
			break;
		}

	}



	std::string fileExtension;

	//Background 폴더에 접근하여 이미지파일 주소들 따오기
	for (int i = 0; i < colorFolders.size(); i++)
	{
		handle = _findfirst((colorFolders[i] + "\\Background\\*.*").c_str(), &fd);
		std::vector<std::string> newImages;

		if (handle == -1)
		{
			printf("there is no File!\n");
		}

		while (true)
		{
			fileName = fd.name;
			tempChar = strrchr(fileName.c_str(), '.');
			if (tempChar == NULL)
			{
				result = _findnext(handle, &fd);
				if (result != 0)
				{
					break;
				}
				continue;
			}

			fileExtension = tempChar;

			for (int i = 0; i < fileExtension.size(); i++)
			{
				fileExtension[i] = tolower(fileExtension[i]);
			}

			if ((fileExtension == ".jpg") || (fileExtension == ".bmp") || (fileExtension == ".png"))
			{
				newImages.push_back(colorFolders[i] + "\\Background\\" + fd.name);
			}

			result = _findnext(handle, &fd);
			if (result != 0)
			{
				break;
			}
		}

		backgroudAddresses_.push_back(newImages);
	}

	//ColorArea 폴더에 접근하여 이미지파일 주소들 따오기
	for (int i = 0; i < colorFolders.size(); i++)
	{
		handle = _findfirst((colorFolders[i] + "\\ColorArea\\*.*").c_str(), &fd);
		std::vector<std::string> newImages;

		if (handle == -1)
		{
			printf("there is no File!\n");
		}

		while (true)
		{
			fileName = fd.name;
			tempChar = strrchr(fileName.c_str(), '.');
			if (tempChar == NULL) continue;

			fileExtension = tempChar;

			if (tempChar == NULL)
			{
				result = _findnext(handle, &fd);
				if (result != 0)
				{
					break;
				}
				continue;
			}

			for (int i = 0; i < fileExtension.size(); i++)
			{
				fileExtension[i] = tolower(fileExtension[i]);
			}

			if ((fileExtension == ".jpg") || (fileExtension == ".bmp") || (fileExtension == ".png"))
			{
				newImages.push_back(colorFolders[i] + "\\ColorArea\\" + fd.name);
			}

			result = _findnext(handle, &fd);
			if (result != 0)
			{
				break;
			}
		}

		colorAreaAddresses_.push_back(newImages);
	}

	_findclose(handle);

	return classes;
}

//새츄레이션 95%와 100%룰 반환하는 함수
std::vector<std::tuple<int, int, int>> ColorDetection::getThresholds(
	std::vector<std::vector<std::string>>* backAddresses, 
	std::vector<std::vector<std::string>>* colorAddresses, 
	std::vector<std::tuple<int, int>>* backThresholds, 
	std::vector<std::tuple<int, int>>* colorThresholds)
{
	auto& backAddresses_ = *backAddresses;
	auto& colorAddresses_ = *colorAddresses;
	auto& backThresholds_ = *backThresholds;
	auto& colorThresholds_ = *colorThresholds;


	cv::Mat frame;
	cv::Mat HSVFrame;
	cv::Vec3b* medianFilteredHSVFramePtr;
	cv::Mat MedianFilteredHSVFrame;
	std::priority_queue<uint8_t, std::vector<uint8_t>, std::less<uint8_t>> backPQ;
	std::priority_queue<uint8_t, std::vector<uint8_t>, std::greater<uint8_t>> colorPQ;
	std::vector<uint8_t> hueValues;

	std::vector<std::tuple<int, int, int>> hueMappings;

	int totalPixel = 0;

	//배경사진 새츄레이션 쓰레숄드 계산
	for (int i = 0; i < backAddresses_.size(); i++)
	{
		std::tuple<int, int> temp = { 0,0 };

		auto tempAddresses = backAddresses_[i];


		for (int ii = 0; ii < tempAddresses.size(); ii++)
		{
			frame = cv::imread(tempAddresses[ii]);
			if (frame.empty() != true)
			{
				std::cout << "processing... : " << tempAddresses[ii] << std::endl;

				cv::cvtColor(frame, HSVFrame, cv::COLOR_BGR2HSV);
				cv::medianBlur(HSVFrame, MedianFilteredHSVFrame, mMedianBlurSize);

				for (int r = 0; r < MedianFilteredHSVFrame.rows; r++)
				{
					for (int c = 0; c < MedianFilteredHSVFrame.cols; c++)
					{
						medianFilteredHSVFramePtr = MedianFilteredHSVFrame.ptr<cv::Vec3b>(r);

						backPQ.push(medianFilteredHSVFramePtr[c][1]);
					}
				}
				totalPixel = totalPixel + MedianFilteredHSVFrame.cols * MedianFilteredHSVFrame.rows;
			}
			else
			{
				std::cout << "processing failed : " << tempAddresses[ii] << std::endl;
			}
		}

		uint8_t top = backPQ.top();

		int sat95 = static_cast<int>(totalPixel * 0.05f);

		for (int sat = 0; sat < sat95; sat++)
		{
			backPQ.pop();
		}

		uint8_t top95 = backPQ.top();

		temp = std::make_tuple(top, top95);

		backThresholds_.push_back(temp);

		while (!backPQ.empty()) backPQ.pop();
	}

	totalPixel = 0;

	//컬러 새츄레이션 쓰레숄드 + 휴 컬러매핑 계산
	for (int i = 0; i < colorAddresses_.size(); i++)
	{
		std::tuple<int, int> temp_int_int = { 0,0 };
		auto tempAddresses = colorAddresses_[i];
		hueValues.clear();

		for (int ii = 0; ii < tempAddresses.size(); ii++)
		{
			frame = cv::imread(tempAddresses[ii]);
			if (frame.empty() != true)
			{
				std::cout << "processing... : " << tempAddresses[ii] << std::endl;

				cv::cvtColor(frame, HSVFrame, cv::COLOR_BGR2HSV);
				cv::medianBlur(HSVFrame, MedianFilteredHSVFrame, mMedianBlurSize);

				for (int r = 0; r < MedianFilteredHSVFrame.rows; r++)
				{
					for (int c = 0; c < MedianFilteredHSVFrame.cols; c++)
					{
						medianFilteredHSVFramePtr = MedianFilteredHSVFrame.ptr<cv::Vec3b>(r);
						auto h_ = medianFilteredHSVFramePtr[c][0];
						auto s_ = medianFilteredHSVFramePtr[c][1];
						auto v_ = medianFilteredHSVFramePtr[c][2];
						colorPQ.push(medianFilteredHSVFramePtr[c][1]);
						hueValues.push_back(medianFilteredHSVFramePtr[c][0]);
						//Hue 컬러 매핑하려고 프라이어리티 큐에 넣음
					}
				}
				totalPixel = totalPixel + HSVFrame.cols * HSVFrame.rows;

			}
			else
			{
				std::cout << "processing failed : " << tempAddresses[ii] << std::endl;
			}
		}

		uint8_t top = colorPQ.top();

		int sat95 = (float)totalPixel * 0.05f;

		for (int sat = 0; sat < sat95; sat++)
		{
			colorPQ.pop();
		}

		uint8_t top95 = colorPQ.top();
		temp_int_int = std::make_tuple(top, top95);
		colorThresholds_.push_back(temp_int_int);


		float hueConfidencePercentage = 0.05f;
		int hueConfidenceCount = static_cast<int>(hueValues.size() * hueConfidencePercentage);
		std::sort(hueValues.begin(), hueValues.end());

		uint8_t huebotomWithConfidence = hueValues[hueConfidenceCount];
		uint8_t hueTopWithConfidence = hueValues[hueValues.size() - hueConfidenceCount];
		bool isZeroInclude = false;

		//탑과 바텀의 값이 127 이상 차이날경우, 사이에 0이 들어있는것으로 판단함
		//따라서 탑과 바텀값을 미리 바꿔놓음
		if (hueTopWithConfidence - huebotomWithConfidence > 0x7F)
		{
			isZeroInclude = true;
			uint8_t tempSwap = hueTopWithConfidence;
			hueTopWithConfidence = huebotomWithConfidence;
			huebotomWithConfidence = tempSwap;
		}

		double hueMid = 0;
		int count = 1;

		for (int i = sat95; i < totalPixel - sat95; i++)
		{
			//0을 포함한 범위일경우, 127을 넘는 값들에는 -255를 더해서 마이너스로 만들어 평균을 계산한다
			if (isZeroInclude && (hueValues[i] > 0x7F))
			{
				hueMid = (hueMid * (count - 1) / (double)count) + (((double)hueValues[i] - 255.0) / count);
			}
			else
			{
				hueMid = (hueMid * (count - 1) / (double)count) + ((double)hueValues[i] / count);
			}
			count++;
		}

		//0을 포함한 범위였을경우, mid의 값이 -면, 원래의 값인 +255로 돌려준다
		if (isZeroInclude && (hueMid <= 0.0))
		{
			hueMid += 255;
		}
		std::tuple<int, int, int> temp_int3;
		temp_int3 = std::make_tuple(hueTopWithConfidence, hueMid, huebotomWithConfidence);
		hueMappings.push_back(temp_int3);


		while (!colorPQ.empty()) colorPQ.pop();
	}


	return hueMappings;

}
