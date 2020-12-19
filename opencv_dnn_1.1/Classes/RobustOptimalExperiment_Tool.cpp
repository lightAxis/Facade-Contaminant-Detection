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
		const float (&NMSThresholdLevels)[], const float (&ConfidenceThresholdLevels)[],
		const eOrthogonalArray& orthArray = L9);

	void DoColorDetection_Experiment(YOLO_v3_DNN* YOLOv3_Module, const float& fixedNMSThreshold, const float& fixedConfidenceThreshold,
		ColorDetection* colorDetection_Module, const int (&hueMarginLevels)[], const int (&medianFilterSizeLevels)[], const int (&downsamplingSizeLevels)[], 
		const eOrthogonalArray& orthArray = L27);

	void DoGrayScaleExperiment(YOLO_v3_DNN* YOLOv3_Module, const float& fixedNMSThreshold, const float& fixedConfidenceThreshold,
		ColorDetection* ColorDetection_Module, const int& fixedHueMargin, const int& fixedMedianFilterSize, const int& fixedDownsamplingSize,
		GrayScaleCalculator* GrayScale_Module, const GrayScaleCalculator::eCalcMethod& calcMethod, const eOrthogonalArray& orthArray = L_None);

	void SaveYoloExperiment_AnswerBox_txt(YOLO_v3_DNN* YOLOv3_Module_, const float &NMSThreshold, const float& confidenceThreshold);
	void SaveAnswer_AreaRatio_txt(const std::string& folder);
	void SaveColorDetection_DetectionBox_txt(ColorDetection* ColorDetection_Module,
		const int& hueMargin, const int& medianFilterSize, const int& downSamplingSize);
};

//욜로 실험 진행하는 함수
void RobustOptimalExperiment::DoYoloExperiment(YOLO_v3_DNN* YOLOv3_Module, 
	const float (&NMSThresholdLevels)[], const float (&confidenceThresholdLevels)[],
	const eOrthogonalArray& orthArray)
{
	auto& YOLOv3_Module_ = *YOLOv3_Module;

	std::string columeVariables[] = { "A","B","AxB(1)","AxB(2)" };
	std::string variableNames[2] = { "A = Nms Threshold","B = Confidence Threshold" };
	float SNRatio_Results[9];
	for (int i = 0; i < 9; i++) SNRatio_Results[i] = 0.0f;

	//float A_s[3] = { 0,0,0 };
	//float B_s[3] = { 0,0,0 };
	//float AxB_1_s[3] = { 0,0,0 };
	//float AxB_2_s[3] = { 0,0,0 };

	float curNMSThreshold = 0.0f;
	float curConfThreshold = 0.0f;

	cv::Mat testImg;
	cv::Mat resultImg;

	std::vector<float> performanceResults;
	//performance_Results.reserve(216);

	std::vector<std::string> imageFileNames;
	std::vector<std::string> TestImageFileAdds = getFilenames("./testImages_resized", &imageFileNames);

	InBoxChecker InboxChecker_Tool;

	//실험횟수 총 9번
	for (int y = 0; y < 9; y++)
	{
		//현재 파라미터 장착
		curNMSThreshold = NMSThresholdLevels[mL9_Array[y][0] - 1];
		curConfThreshold = confidenceThresholdLevels[mL9_Array[y][1] - 1];
		//욜로 준비
		YOLOv3_Module_.SetNMSThreshold(curNMSThreshold);
		YOLOv3_Module_.SetConfidenceThreshold(curConfThreshold);
		//이미지 파일 읽어들인다음 한번 테스트 및 IoUc 계산
		int count_ = TestImageFileAdds.size();
		//결과값 저장 벡터 초기화
		performanceResults.clear();

		for (int i = 0; i < count_; i++)
		{
			InboxChecker_Tool.ClearBox();

			testImg = cv::imread(TestImageFileAdds[i]);

			auto startTime = std::chrono::high_resolution_clock::now();
			YOLOv3_Module_.PassThrough(&testImg);
			YOLOv3_Module_.GetObjectRects(&InboxChecker_Tool);
			auto endTime = std::chrono::high_resolution_clock::now();
			std::cout << std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() << std::endl;
			std::vector<InBoxChecker::BoxInfo> tempInfos = InboxChecker_Tool.GetBoxes(InBoxChecker::YOLOv3);
			std::vector<cv::Rect> detections;
			for (int i = 0; i < tempInfos.size(); i++)
			{
				detections.push_back(tempInfos[i].Box);
			}

			std::string boxInfo_address = imageFileNames[i].substr(0, imageFileNames[i].length() - 4);
			boxInfo_address = "./testImages_resized\\" + boxInfo_address + ".txt";
			std::vector<cv::Rect> answers = getYOLOBoxes(boxInfo_address, testImg.cols, testImg.rows);

			float IoUc = compare_YOLO_IoUc(answers, detections, testImg.cols, testImg.rows);
			performanceResults.push_back(IoUc);

			std::cout << std::to_string(y) << "_" << std::to_string(i) << " : IoUc = " << std::to_string(IoUc) << std::endl;
		}

		//SN비 계산 후 저장
		double SN_temp = 0.0;
		for (int i = 0; i < performanceResults.size(); i++)
		{
			SN_temp += pow(1.0 - performanceResults[i], 2.0);
		}
		SN_temp = -10.0 * log10(SN_temp / performanceResults.size());

		SNRatio_Results[y] = (float)SN_temp;

		std::cout << std::to_string(y) << " : SN ratio = " << std::to_string(SN_temp) << std::endl;
	}

	//텍스트 파일로 출력
	std::ofstream writeT;
	writeT.open("G_YOLO_Results.txt");
	writeT << "YOLOv3 RobustOptimal Experiment Results\n";
	writeT << variableNames[0] << "," << std::to_string(NMSThresholdLevels[0]) << "," << std::to_string(NMSThresholdLevels[1]) << "," << std::to_string(NMSThresholdLevels[2]) << "\n";
	writeT << variableNames[1] << "," << std::to_string(confidenceThresholdLevels[0]) << "," << std::to_string(confidenceThresholdLevels[1]) << "," << std::to_string(confidenceThresholdLevels[2]) << "\n";
	writeT << "Rows," << columeVariables[0] << "," << columeVariables[1] << "," + columeVariables[2] << "," + columeVariables[3] << "\n";
	std::string writeLine_ = "";

	for (int i = 0; i < 9; i++)
	{
		writeLine_ = "";
		writeLine_ = writeLine_ + std::to_string(i + 1) + "," + std::to_string(mL9_Array[i][0]) + "," + std::to_string(mL9_Array[i][1]) + ","
			+ std::to_string(mL9_Array[i][2]) + "," + std::to_string(mL9_Array[i][3]) + "," + std::to_string(SNRatio_Results[i]) + "\n";
		writeT << writeLine_;
	}

	writeT.close();

	std::cout << "done!" << std::endl;

};

//컬러 디텍션 모듈 실험을 진행하는 함수
void RobustOptimalExperiment::DoColorDetection_Experiment(
	YOLO_v3_DNN* YOLOv3_Module, const float& fixedNMSThreshold, const float& fixedConfidenceThreshold,
	ColorDetection* colorDetection_Module, const int (&hueMarginLevels)[], const int (&medianFilterSizeLevels)[], const int (&downsamplingSizeLevels)[],
	const eOrthogonalArray& orthArray)
{
	auto& YOLOv3_Module_ = *YOLOv3_Module;
	auto& ColorDetection_Module_ = *colorDetection_Module;

	std::string columeVariables[10] = { "C","D","CxD(1)","CxD(2)","E","CxE(1)","CxE(2)","DxE(1)","","DxE(2)" };
	std::string variableNames[3] = { "C = Hue margin","D = MedianFilter Size","E = Downsampling Size" };
	float SN_Ratio_Results[27];
	for (int i = 0; i < 27; i++) SN_Ratio_Results[i] = 0.0f;
	cv::Mat HSVedFrame;

	int curHueMargin = 0;
	int curMedianFiterSize = 0;
	int curDownSamplingSize = 0;;

	cv::Mat testImg;
	cv::Mat resultImg;

	std::vector<float> performanceResults;
	//performance_Results.reserve(216);

	std::vector<std::string> imageFileNames;
	std::vector<std::string> TestImageFileAdds = getFilenames("./testImages_resized", &imageFileNames);

	InBoxChecker InboxChecker_Tool;

	YOLOv3_Module_.SetConfidenceThreshold(fixedConfidenceThreshold);
	YOLOv3_Module_.SetNMSThreshold(fixedNMSThreshold);

	//컬러 디텍션 모듈 초기화
	ColorDetection_Module_.ClearColorDetectionRange();
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_1_B1_M1", hueMarginLevels[0]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_1_B1_M2", hueMarginLevels[1]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_1_B1_M3", hueMarginLevels[2]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_1_B2_M1", hueMarginLevels[0]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_1_B2_M2", hueMarginLevels[1]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_1_B2_M3", hueMarginLevels[2]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_1_B3_M1", hueMarginLevels[0]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_1_B3_M2", hueMarginLevels[1]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_1_B3_M3", hueMarginLevels[2]);

	ColorDetection_Module_.MakeColorDetection("autoColorParameter_2_B1_M1", hueMarginLevels[0]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_2_B1_M2", hueMarginLevels[1]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_2_B1_M3", hueMarginLevels[2]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_2_B2_M1", hueMarginLevels[0]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_2_B2_M2", hueMarginLevels[1]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_2_B2_M3", hueMarginLevels[2]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_2_B3_M1", hueMarginLevels[0]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_2_B3_M2", hueMarginLevels[1]);
	ColorDetection_Module_.MakeColorDetection("autoColorParameter_2_B3_M3", hueMarginLevels[2]);

	//27개의 콜럼으로 진행됨
	for (int y = 0; y < 27; y++)
	{
		curHueMargin = hueMarginLevels[mL27_Array[y][0] - 1];
		curMedianFiterSize = medianFilterSizeLevels[mL27_Array[y][1] - 1];
		curDownSamplingSize = downsamplingSizeLevels[mL27_Array[y][4] - 1];

		//탐지된 파일 개수만큼 반복
		int count = TestImageFileAdds.size();
		//결과값 저장 벡터 초기화
		performanceResults.clear();

		for (int i = 0; i < count; i++)
		{
			//인박스 체커 클리어
			InboxChecker_Tool.ClearBox();
			testImg = cv::imread(TestImageFileAdds[i]);

			//미리 저장한 욜로모듈결과값을 가져옴
			getYOLOv3DetectionBox(TestImageFileAdds[i], &InboxChecker_Tool);

			//정답 에어리어 값 불러옴
			float trueAreaRatio = getColorSectionAreaRatioAnswer(TestImageFileAdds[i]);
			std::vector<cv::Rect> trueBoxes = getColorBoxes(TestImageFileAdds[i], testImg.cols, testImg.rows);


			//5개로 쪼개지고, 마지막 녀석은 .jpg가 붙어있으니 주의
			//순서대로 시편보드색깔, 오염물타입, 높이단계, 밝기단계, 회전단계임
			//오염물타입이 ABC로 되어있고, 높이단계는 H-가 붙어있고,밝기단계는 B-가 붙어있음
			std::vector<std::string> split_filename = splitStringByChar(imageFileNames[i], '_');

			//1~3까지
			int marginLevel = mL27_Array[y][0];
			//1~2까지
			int boardColor = std::atoi(split_filename[0].c_str());
			//1~3까지
			int brightnessLevel = (split_filename[3])[2] - '0';

			int colorRangeNum = (boardColor - 1) * 9 + (brightnessLevel - 1) * 3 + marginLevel - 1;

			//알맞은 컬러 디텍션 레인지에 통과시킴
			auto downSamplingSize_enum = static_cast<ColorDetection::eDownSamplingSizes>(curDownSamplingSize);
			ColorDetection_Module_.SetDownSamplingSize(downSamplingSize_enum);
			auto medianFiltersize_enum = static_cast<ColorDetection::eMedianBlurSizes>(curMedianFiterSize);
			ColorDetection_Module_.SetMedianBlurSize(medianFiltersize_enum);

			auto startTime = std::chrono::high_resolution_clock::now();
			int detectedPixels = ColorDetection_Module_.PushThroughImage(testImg, &HSVedFrame, &InboxChecker_Tool, colorRangeNum);
			auto endTime = std::chrono::high_resolution_clock::now();

			float DetectedPixelRatio = (float)detectedPixels / (testImg.cols * testImg.rows);
			auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

			cv::waitKey(1);

			float IoUa = compareColorDetection_errIoUa(trueBoxes, &InboxChecker_Tool, trueAreaRatio, DetectedPixelRatio, elapsedTime, testImg.cols, testImg.rows);//정답 면적비 계산해야함
			std::cout << "Col: " << y + 1 << "/img(" << i << "): " << TestImageFileAdds[i] << "/ errIoUa: " << IoUa << std::endl;
			performanceResults.push_back(IoUa);
		}

		//SN비 계산 후 저장
		double SN_temp = 0.0;
		for (int i = 0; i < performanceResults.size(); i++)
		{
			SN_temp += pow(performanceResults[i], 2.0);
		}
		SN_temp = -10.0 * log10(SN_temp / performanceResults.size());
		SN_Ratio_Results[y] = (float)SN_temp;
		std::cout << "SN Ratio( " << y + 1 << "): " << SN_temp << std::endl;
	}

	//텍스트 파일로 출력
	std::ofstream writeT;
	writeT.open("G_COLOR_Results.txt");
	writeT << "Color Detection RobustOptimal Experiment Results\n";
	writeT << variableNames[0] << "," << hueMarginLevels[0] << "," << hueMarginLevels[1] << "," << hueMarginLevels[2] << "\n";
	writeT << variableNames[1] << "," << medianFilterSizeLevels[0] << "," << medianFilterSizeLevels[1] << "," << medianFilterSizeLevels[2] << "\n";
	writeT << variableNames[2] << "," << downsamplingSizeLevels[0] << "," << downsamplingSizeLevels[1] << "," << downsamplingSizeLevels[2] << "\n";
	writeT << "Variables," << columeVariables[0] << "," << columeVariables[1] << "," + columeVariables[2] << "," + columeVariables[3]
		<< "," + columeVariables[4] << "," + columeVariables[5] << "," + columeVariables[6] << "," + columeVariables[7]
		<< "," + columeVariables[8] << "," + columeVariables[9] << "\n";

	std::string writeLine_ = "";
	for (int i = 0; i < 27; i++)
	{
		writeLine_ = "";
		writeLine_ = writeLine_ + std::to_string(i + 1) + "," + std::to_string(mL27_Array[i][0]) + "," + std::to_string(mL27_Array[i][1])
			+ "," + std::to_string(mL27_Array[i][2]) + "," + std::to_string(mL27_Array[i][3]) + "," + std::to_string(mL27_Array[i][4])
			+ "," + std::to_string(mL27_Array[i][5]) + "," + std::to_string(mL27_Array[i][6]) + "," + std::to_string(mL27_Array[i][7])
			+ "," + std::to_string(mL27_Array[i][8]) + "," + std::to_string(mL27_Array[i][9]) + "," + std::to_string(mL27_Array[i][10])
			+ "," + std::to_string(mL27_Array[i][11]) + "," + std::to_string(mL27_Array[i][12]) + "," + std::to_string(SN_Ratio_Results[i]) + "\n";
		writeT << writeLine_;
	}
	writeT.close();

};

//그레이스케일 모듈 실험을 진행하는 함수
void RobustOptimalExperiment::DoGrayScaleExperiment(
	YOLO_v3_DNN* YOLOv3_Module, const float& fixedNMSThreshold, const float& fixedConfidenceThreshold,
	ColorDetection* colorDetection_Module, const int& fixedHueMargin, const int& fixedMedianFilterSize, const int& fixedDownsamplingSize,
	GrayScaleCalculator* GrayScale_Module, const GrayScaleCalculator::eCalcMethod& calcMethod, const eOrthogonalArray& orthArray)
{
	auto& YOLOv3_Module_ = *YOLOv3_Module;
	auto& colorDetection_Module_ = *colorDetection_Module;
	auto& GrayScale_Module_ = *GrayScale_Module;


	//어떤 컬러 스페이스를 쓸지 정하기
	GrayScaleCalculator::eCalcMethod currcalcMethod = calcMethod;


	std::vector<std::string> imagefile_names;
	std::vector<std::string> TestImageFileAdds = getFilenames("./testImages_resized", &imagefile_names);

	InBoxChecker InboxChecker_tool;
	int count_ = TestImageFileAdds.size();
	std::vector<std::vector<float>> answerMat;

	for (int board = 1; board <= 2; board++)
	{
		for (int batch = 1; batch <= 3; batch++)
		{
			for (int height = 1; height <= 2; height++)
			{
				for (int bright = 1; bright <= 3; bright++)
				{
					std::vector<float> line;
					line.push_back(board);
					line.push_back(batch);
					line.push_back(height);
					line.push_back(bright);
					line.push_back(0.0f);

					answerMat.push_back(line);
				}
			}
		}
	}


	GrayScale_Module_.ClearReferences();

	GrayScale_Module_.SetCalcMethod(currcalcMethod);
	GrayScale_Module_.AddReference("./grayscale_references/1_B-1_ref.jpg");
	GrayScale_Module_.AddReference("./grayscale_references/1_B-2_ref.jpg");
	GrayScale_Module_.AddReference("./grayscale_references/1_B-3_ref.jpg");
	GrayScale_Module_.AddReference("./grayscale_references/2_B-1_ref.jpg");
	GrayScale_Module_.AddReference("./grayscale_references/2_B-2_ref.jpg");
	GrayScale_Module_.AddReference("./grayscale_references/2_B-3_ref.jpg");

	for (int i = 0; i < TestImageFileAdds.size(); i++)
	{
		InboxChecker_tool.ClearBox();

		getYOLOv3DetectionBox(TestImageFileAdds[i], &InboxChecker_tool);
		getColorDetectionDetectionBox(TestImageFileAdds[i], &InboxChecker_tool);

		GrayScale_Module_.SetCalcMethod(currcalcMethod);

		std::vector<std::string> splitedFilename = splitStringByChar(imagefile_names[i], '_');
		//보드 판 색깔 1~2까지
		int boardColor = std::atoi(splitedFilename[0].c_str());
		//오염물 배치 타입 1~2 까지
		int batchType = splitedFilename[1][0] - 'A' + 1;
		//촬영 높이 1~2까지
		int cameraHeight = splitedFilename[2][2] - '0';
		//밝기 1~3까지
		int brightnessLevel = splitedFilename[3][2] - '0';
		//방향 정보 1~4까지
		int directionLevel = std::atoi(splitedFilename[4].c_str());

		//해당 값이 들어갈 index 계산
		int valueIndex = (boardColor - 1) * 18 + (batchType - 1) * 6 + (cameraHeight - 1) * 3 + (brightnessLevel - 1);

		//모듈의 결과값 계산
		cv::Mat testImg = cv::imread(TestImageFileAdds[i]);
		float detectionValue = GrayScale_Module_.CalcGrayScale(testImg, &InboxChecker_tool);

		//해당 이미지에 맞는 레퍼런스의 index 계산
		int referenceIndex = (boardColor - 1) * 3 + (brightnessLevel - 1);
		//맞는 레퍼런스의 값 가져옴
		float referenceValue = (GrayScale_Module_.GetReferenceAt(referenceIndex)).AverageBrightness;

		float differenceAverageBrightness = referenceValue - detectionValue;
		answerMat[valueIndex][4] = answerMat[valueIndex][4] + differenceAverageBrightness;
		std::cout << imagefile_names[i] << std::endl;
		std::cout << "(" + std::to_string(i + 1) << "/" << count_ << "), diff : " << differenceAverageBrightness << std::endl;
	}
	//결과값들을 전부 4로 나누기(평균때문에 4장씩 찍었던거이므로)
	for (int i = 0; i < answerMat.size(); i++)
	{
		answerMat[i][4] = answerMat[i][4] / 4.0f;
	}

	//결과값을 csv 파일로 저장하기
	std::ofstream writeT;
	std::string resultName = "G_Gray_result";
	switch (currcalcMethod)
	{
	case GrayScaleCalculator::RGB_Luminance:
	{
		resultName = resultName + "_Y_YUV";
		break;
	}
	case GrayScaleCalculator::HSV_Value:
	{
		resultName = resultName + "_V_HSV";
		break;
	}
	case GrayScaleCalculator::HSL_Lightness:
	{
		resultName = resultName + "_L_HSL";
		break;
	}
	case GrayScaleCalculator::CIELab_Lightness:
	{
		resultName = resultName + "_L_CIELab";
		break;
	}
	default:
		break;
	}
	resultName = resultName + ".txt";

	writeT.open(resultName);
	writeT << "Grayscale Experiment Results\n";
	writeT << "Board,Batch,Height,Brightness\n";

	std::string writeLine = "";
	for (int i = 0; i < answerMat.size(); i++)
	{
		writeLine = std::to_string(answerMat[i][0]) + "," + std::to_string(answerMat[i][1])
			+ "," + std::to_string(answerMat[i][2]) + "," + std::to_string(answerMat[i][3]) + ","
			+ std::to_string(answerMat[i][4]) + "\n";
		writeT << writeLine;
	}
	writeT.close();


};


//특정 폴더에 들어있는 모든 jpg 파일을 읽어들이는 함수.
std::vector<std::string> RobustOptimalExperiment::getFilenames(const std::string& folderAddress, std::vector<std::string>* fileNames)
{
	auto& fileNames_ = *fileNames;

	std::string searching = folderAddress + "\\" + "*.jpg";
	std::string fileExtension;
	std::vector<std::string> imageNames;

	_finddata_t fd;
	intptr_t handle;
	int result;
	handle = _findfirst(searching.c_str(), &fd);

	std::string fileName;
	const char* temp;

	//내부에 원하는 클래스 폴더 탐지
	if (handle == -1)
	{
		printf("there is no File!\n");
		return imageNames;
	}

	while (true)
	{
		fileName = fd.name;
		std::cout << "getImage : " << fileName << std::endl;

		imageNames.push_back(folderAddress + "\\" + fd.name);
		fileNames_.push_back(fd.name);

		result = _findnext(handle, &fd);
		if (result != 0)
		{
			break;
		}
	}

	return imageNames;
}

//---------------------------욜로 관련 서브 함수----------------------

//욜로 박스 데이터로부터 cv::Rect로의 변환 과정
std::vector<cv::Rect> RobustOptimalExperiment::getYOLOBoxes(const std::string& path, const int& frameCols, const int& frameRows)
{
	std::ifstream readT(path);
	std::string getSplit[5] = { "","","","","" };
	float getXYWH[4] = { 0.0f,0.0f,0.0f,0.0f };
	//char line[100];
	std::string line = "";
	cv::Rect answerRect;
	std::vector<cv::Rect> answerRects;

	while (std::getline(readT, line))
	{
		for (int i = 0; i < 5; i++) getSplit[i] = "";
		int split_i = 0;
		for (int i = 0; i < line.length(); i++)
		{
			if (line[i] == ' ')
			{
				split_i++;
				continue;
			}
			else
			{
				getSplit[split_i] = getSplit[split_i] + line[i];
			}
		}
		for (int i = 1; i < 5; i++) getXYWH[i - 1] = atof(getSplit[i].c_str());
		getXYWH[0] = getXYWH[0] * frameCols;
		getXYWH[1] = getXYWH[1] * frameRows;
		getXYWH[2] = getXYWH[2] * frameCols;
		getXYWH[3] = getXYWH[3] * frameRows;
		answerRect = cv::Rect(getXYWH[0] - getXYWH[2] / 2, getXYWH[1] - getXYWH[3] / 2, getXYWH[2], getXYWH[3]);
		answerRects.push_back(answerRect);
	}

	readT.close();

	return answerRects;
}

//YOLOv3 모듈에서 쓰는 평가 지표 계산
float RobustOptimalExperiment::compare_YOLO_IoUc(const std::vector<cv::Rect>& answerBoxes, const std::vector<cv::Rect>& detectedBoxes
	, const int& frameCols, const int& frameRows)
{

	if (answerBoxes.size() == 0) //만약 true box 개수가 0일경우
	{
		if (detectedBoxes.size() == 0)
		{
			return 1.0f;
		}
		else
		{
			return 0.0f;
		}
	}
	else//전체 영역의 iou계산 시작
	{
		int total_intersect_pixels = 0;
		int total_pixels = 0;
		cv::Point pt;
		bool answer_inside = false;
		bool detect_inside = false;

		int detected_box_index = -1;

		for (int r = 0; r < frameRows; r++)
		{
			for (int c = 0; c < frameCols; c++)
			{
				detected_box_index = -1;
				answer_inside = false;
				detect_inside = false;
				pt = cv::Point(c, r);

				for (int i = 0; i < answerBoxes.size(); i++)
				{
					if (answerBoxes[i].contains(pt) == true)
					{
						answer_inside = true;
						detected_box_index = i;
						break;
					}
				}

				for (int i = 0; i < detectedBoxes.size(); i++)
				{
					if (detectedBoxes[i].contains(pt) == true)
					{
						detect_inside = true;
						break;
					}
				}

				if (answer_inside)
				{
					if (detect_inside)
					{
						total_intersect_pixels++;
						total_pixels++;
						continue;
					}
					else
					{
						total_pixels++;
						continue;
					}
				}
				else if (detect_inside)
				{
					total_pixels++;
					continue;
				}
				else
				{
					continue;
				}
			}
		}

		float IoU = (float)total_intersect_pixels / total_pixels;//전체 영역 대상으로 한 IoU


		// detection box가 잘못된게 있는지 검사 시작
		std::vector<bool> detectionboxes_isboxnormal;
		for (int i = 0; i < detectedBoxes.size(); i++)
		{
			detectionboxes_isboxnormal.push_back(false);
		}
		std::vector<int> answerboxes_intersectioncount;
		for (int i = 0; i < answerBoxes.size(); i++)
		{
			answerboxes_intersectioncount.push_back(0);
		}

		int detectionbox_intersection_count = 0;
		bool isboxhaveAtleast05IoU = false;

		for (int i = 0; i < detectedBoxes.size(); i++)
		{
			detectionbox_intersection_count = 0;
			isboxhaveAtleast05IoU = false;
			for (int j = 0; j < answerBoxes.size(); j++)
			{
				cv::Rect intersectBox = detectedBoxes[i] & answerBoxes[j];
				if (intersectBox.area() < 0.001f)
				{
					continue;
				}
				detectionbox_intersection_count++;

				float boxIoU = (float)intersectBox.area() / (float)(detectedBoxes[i].area() + answerBoxes[j].area() - intersectBox.area());
				if (boxIoU >= 0.5f)
				{
					isboxhaveAtleast05IoU = true;

					if (answerboxes_intersectioncount[j] == 0)
					{
						answerboxes_intersectioncount[j]++;
					}
					else
					{
						detectionbox_intersection_count = 100;
					}
				}


			}

			if ((detectionbox_intersection_count == 1) && (isboxhaveAtleast05IoU == true))
			{
				detectionboxes_isboxnormal[i] = true;
			}
			else
			{
				detectionboxes_isboxnormal[i] = false;
			}
		}


		int trueCount = answerBoxes.size();
		int detectCount = detectedBoxes.size();
		int correctCount = 0;
		for (int i = 0; i < detectionboxes_isboxnormal.size(); i++)
		{
			if (detectionboxes_isboxnormal[i] == true) correctCount++;
		}

		std::cout << "IoU : " << std::to_string(IoU) << "/ TC : " << std::to_string(trueCount) << "/DC : " << std::to_string(detectCount) << "/CC : " << std::to_string(correctCount) << std::endl;
		float IoUc = IoU * std::powf((float)detectCount / (trueCount + 2), fabsf(((float)trueCount - detectCount) / trueCount));
		//만약 디텍션 카운트가 0 이상이라면, correction도 반영, 디텍션 카운트가 0이면 자동으로 값도 0이 됨.
		if (detectCount > 0)
		{
			IoUc = IoUc * std::powf((float)correctCount / (detectCount + 2), fabsf(((float)detectCount - correctCount) / detectCount));
		}

		return IoUc;
	}
}


//--------------------------컬러 모듈 관련 서브 함수------------------------

//시편의 정답 영역의 넓이 읽어들이는 함수
//path_에는 해당 사진의 전체 경로만 주면 됨
float RobustOptimalExperiment::getColorSectionAreaRatioAnswer(const std::string& path)
{
	std::ifstream readT;
	std::string line = "";
	float trueAreaRatio = 0.0f;

	std::string AreaPath = path.substr(0, path.length() - 4) + "_Area_ratio_Answer.txt";

	readT.open(AreaPath);

	while (std::getline(readT, line))
	{
		trueAreaRatio = atof(line.c_str());
	}

	readT.close();

	return trueAreaRatio;
}

//시편의 정답 박스 읽어들이는 함수
//path_에는 해당 사진의 전체 경로만 주면 됨
std::vector<cv::Rect> RobustOptimalExperiment::getColorBoxes(const std::string& path, const int& frameCols, const int& frameRows)
{
	std::string getSplit[5] = { "","","","","" };
	float getXYWH[4] = { 0.0f,0.0f,0.0f,0.0f };
	//char line[100];
	std::string line = "";
	cv::Rect answerRect;
	std::vector<cv::Rect> answerRects;

	//컬러 영역의 박스 주소로 수정
	std::string BoxPath = path.substr(0, path.length() - 4);
	BoxPath = BoxPath + "_AreaBox.txt";

	std::ifstream readT(BoxPath);

	while (std::getline(readT, line))
	{
		for (int i = 0; i < 5; i++) getSplit[i] = "";
		int split_i = 0;
		for (int i = 0; i < line.length(); i++)
		{
			if (line[i] == ' ')
			{
				split_i++;
				continue;
			}
			else
			{
				getSplit[split_i] = getSplit[split_i] + line[i];
			}
		}
		for (int i = 1; i < 5; i++) getXYWH[i - 1] = atof(getSplit[i].c_str());
		getXYWH[0] = getXYWH[0] * frameCols;
		getXYWH[1] = getXYWH[1] * frameRows;
		getXYWH[2] = getXYWH[2] * frameCols;
		getXYWH[3] = getXYWH[3] * frameRows;
		answerRect = cv::Rect(getXYWH[0] - getXYWH[2] / 2, getXYWH[1] - getXYWH[3] / 2, getXYWH[2], getXYWH[3]);
		answerRects.push_back(answerRect);
	}
	readT.close();

	return answerRects;

}

//시편의 컬러 디텍션 IoUp를 계산하여 반환하는 함수
float RobustOptimalExperiment::compareColorDetection_errIoUa(const std::vector<cv::Rect>& answerBoxes, InBoxChecker* currentInboxChecker,
	const float& answerAreaRatio, const float& detectedAreaRatio, const long long& elapsedTime, const int& frameCols, const int& frameRows)
{
	auto& currentInboxChecker_ = *currentInboxChecker;

	std::vector<InBoxChecker::BoxInfo> resultColorBoxeinfos = currentInboxChecker_.GetBoxes(InBoxChecker::ColorDetection);
	std::vector<cv::Rect> detectedBoxes;
	for (int i = 0; i < resultColorBoxeinfos.size(); i++)
	{
		detectedBoxes.push_back(resultColorBoxeinfos[i].Box);
	}

	int intersectPixels = 0;
	int totalPixels = 0;
	cv::Point pt;
	bool answerInside = false;
	bool detectInside = false;

	if (elapsedTime > 32000) { return 1.0f; }

	float IoU = 0.0f;

	if (answerBoxes.size() == 0)
	{
		if (detectedBoxes.size() == 0)
		{
			IoU = 1;
			return 0.0f;
		}
	}

	for (int r = 0; r < frameRows; r++)
	{
		for (int c = 0; c < frameCols; c++)
		{
			answerInside = false;
			detectInside = false;

			pt = cv::Point(c, r);

			for (int i = 0; i < answerBoxes.size(); i++)
			{
				if (answerBoxes[i].contains(pt) == true)
				{
					answerInside = true;
					break;
				}
			}

			for (int i = 0; i < detectedBoxes.size(); i++)
			{
				if (detectedBoxes[i].contains(pt) == true)
				{
					detectInside = true;
					break;
				}
			}

			if (answerInside)
			{
				if (detectInside)
				{
					intersectPixels++;
					totalPixels++;
					continue;
				}
				else
				{
					totalPixels++;
					continue;
				}
			}
			else if (detectInside)
			{
				totalPixels++;
				continue;
			}
			else
			{
				continue;
			}
		}
	}

	if (answerBoxes.size() == 0)
	{
		int tempPixels = frameCols * frameRows;
		IoU = (float)(tempPixels - totalPixels) / tempPixels;
	}
	else
	{
		IoU = (float)intersectPixels / totalPixels;
	}

	float AAR = 0.0f;
	if (answerAreaRatio < 0.001)
	{
		if (detectedAreaRatio < 0.001) AAR = 1.0f;
		else AAR = 0.0f;
	}
	else
	{
		AAR = (1 - std::fabsf((answerAreaRatio - detectedAreaRatio) / (answerAreaRatio)));
		if (AAR < 0) AAR = 0.0f;
	}


	float IoUa = IoU * AAR;

	float errIoUa = (1 - IoUa);

	return errIoUa;
}

//--------------------------------------------------견실설계 발버둥
void doTest()
{
	int heights = 1;
	int brightnesses = 1;

	cv::Mat frame;
	cv::Rect answerRect;
	cv::Rect detectedRect;
	float GSAnswers[6] = { 0,0,0,0,0,0 };
	float NmsTs[3] = { 0.1,0.5,0.9 };
	float ConfTs[3] = { 0.5,0.6,0.7 };

	std::ifstream readT;
	std::ofstream writeT;
	writeT.open("GResults.txt");
	std::string getSplit[5] = { "","","","","" };
	float getXYWH[4] = { 0.0f,0.0f,0.0f,0.0f };
	char line[100];


	//라인 한줄 쓰기

	std::string Writeline = "," + std::to_string(GSAnswers[0]) + "," + std::to_string(GSAnswers[1]) + "," + std::to_string(GSAnswers[2])
		+ "," + std::to_string(GSAnswers[3]) + "," + std::to_string(GSAnswers[4]) + "," + std::to_string(GSAnswers[5]);
	writeT << Writeline;
	writeT << "\n";

	std::cout << "File Write : " << Writeline << std::endl;

	writeT.close();

}

//특정 캐릭터 하나로 끊어진 스트링의 벡터를 반환하는 함수
std::vector<std::string> RobustOptimalExperiment::splitStringByChar(const std::string& str, const char& delimiter)
{
	int strLen = str.length();
	std::vector<std::string> stringResults;
	char temp = ' ';
	std::string oneLine = "";

	for (int i = 0; i < strLen; i++)
	{
		temp = str[i];
		if (temp == delimiter)
		{
			stringResults.push_back(oneLine);
			oneLine = "";
			continue;
		}
		else
		{
			oneLine = oneLine + temp;
		}
	}

	stringResults.push_back(oneLine);

	return stringResults;
}

//욜로 모듈의 실험 결과를 저장하는 함수
void RobustOptimalExperiment::SaveYoloExperiment_AnswerBox_txt(YOLO_v3_DNN* YOLOv3_Module, const float& NMSThreshold, const float& confidenceThreshold)
{
	auto& YOLOv3_Module_ = *YOLOv3_Module;
	std::vector<std::string> imageFileNames;
	std::vector<std::string> TestImageFileAdds = getFilenames("./testImages_resized", &imageFileNames);

	InBoxChecker InboxChecker_tool;

	YOLOv3_Module_.SetConfidenceThreshold(confidenceThreshold);
	YOLOv3_Module_.SetNMSThreshold(NMSThreshold);

	int count = imageFileNames.size();

	for (int i = 0; i < count; i++)
	{
		InboxChecker_tool.ClearBox();
		cv::Mat tempImg = cv::imread(TestImageFileAdds[i]);
		YOLOv3_Module_.PassThrough(&tempImg);
		YOLOv3_Module_.GetObjectRects(&InboxChecker_tool);

		std::ofstream writeT;
		std::string resultSaveAdd = TestImageFileAdds[i].substr(0, TestImageFileAdds[i].length() - 4) + "_YOLOv3_Detection_Results.txt";

		auto boxes = InboxChecker_tool.GetBoxes(InBoxChecker::YOLOv3);

		writeT.open(resultSaveAdd);
		for (int j = 0; j < boxes.size(); j++)
		{
			writeT << boxes[j].Name << "," << boxes[j].Box.x << "," << boxes[j].Box.y << "," << boxes[j].Box.width << "," << boxes[j].Box.height << "\n";
		}
		writeT.close();

		std::cout << boxes.size() << " boxes saved (" << i + 1 << "/" << count << ")" << "\n";
	}


}

//욜로모듈의 디텍션 결과 세이브 한것을 로드한다
std::vector<cv::Rect> RobustOptimalExperiment::getYOLOv3DetectionBox(const std::string& imgAdded, InBoxChecker* inboxCheckerTool)
{
	auto& inboxCheckerTool_ = *inboxCheckerTool;

	std::string path = imgAdded.substr(0, imgAdded.length() - 4) + "_YOLOv3_Detection_Results.txt";
	std::ifstream readT(path);
	std::vector<std::string> splittedstr;
	std::string line = "";
	int XYWH[4] = { 0,0,0,0 };
	cv::Rect answerRect;
	std::vector<cv::Rect> answerRects;

	while (std::getline(readT, line))
	{
		splittedstr = splitStringByChar(line, ',');
		XYWH[0] = std::atoi(splittedstr[1].c_str());
		XYWH[1] = std::atoi(splittedstr[2].c_str());
		XYWH[2] = std::atoi(splittedstr[3].c_str());
		XYWH[3] = std::atoi(splittedstr[4].c_str());
		answerRect = cv::Rect(XYWH[0], XYWH[1], XYWH[2], XYWH[3]);

		answerRects.push_back(answerRect);
		inboxCheckerTool_.AddBox(answerRect, InBoxChecker::YOLOv3, splittedstr[0]);
	}

	readT.close();
	return answerRects;
}

//컬러디텍션모듈의 디텍션 결과를 가져온다
std::vector<cv::Rect> RobustOptimalExperiment::getColorDetectionDetectionBox(const std::string& imgAdded, InBoxChecker* inboxCheckerTool)
{
	auto& inboxCheckerTool_ = *inboxCheckerTool;

	std::string path = imgAdded.substr(0, imgAdded.length() - 4) + "_ColorDetection_Detection_Results.txt";
	std::ifstream readT(path);
	std::vector<std::string> splittedStr;
	std::string line = "";
	int XYWH[4] = { 0,0,0,0 };
	cv::Rect answerRect;
	std::vector<cv::Rect> answerRects;

	while (std::getline(readT, line))
	{
		splittedStr = splitStringByChar(line, ',');
		XYWH[0] = std::atoi(splittedStr[1].c_str());
		XYWH[1] = std::atoi(splittedStr[2].c_str());
		XYWH[2] = std::atoi(splittedStr[3].c_str());
		XYWH[3] = std::atoi(splittedStr[4].c_str());
		answerRect = cv::Rect(XYWH[0], XYWH[1], XYWH[2], XYWH[3]);

		answerRects.push_back(answerRect);
		inboxCheckerTool_.AddBox(answerRect, InBoxChecker::ColorDetection, splittedStr[0]);
	}

	readT.close();
	return answerRects;
}

//컬러디텍션 모듈의 정답지중 area_ratio 정보를 저장한다
void RobustOptimalExperiment::SaveAnswer_AreaRatio_txt(const std::string& folder)
{
	std::string originalPath = folder;

	std::vector<std::string> fileNames;
	std::vector<std::string> imageAddresses = getFilenames(originalPath, &fileNames);

	cv::Mat originalFrame;
	cv::Vec3b* originalFramePtr;

	int count = imageAddresses.size();
	int pixelCount = 0;
	float ratioResult = 0.0f;

	if (count > 0)
	{
		for (int i = 0; i < count; i++)
		{
			pixelCount = 0;
			originalFrame = cv::imread(imageAddresses[i]);

			for (int y = 0; y < originalFrame.rows; y++)
			{
				for (int x = 0; x < originalFrame.cols; x++)
				{
					originalFramePtr = originalFrame.ptr<cv::Vec3b>(y);
					//완전히 B255G0R0인 픽셀의 개수만을 새서 비율을 반환함
					if (originalFramePtr[x][0] == 53)
					{
						if (originalFramePtr[x][1] == 57)
						{
							if (originalFramePtr[x][2] == 229)
							{
								pixelCount++;
							}
						}
					}
				}
			}

			ratioResult = (float)pixelCount / (originalFrame.cols * originalFrame.rows);

			std::ofstream writeT;
			std::string output_address = imageAddresses[i].substr(0, imageAddresses[i].length() - 4) + "_Area_ratio_Answer.txt";
			writeT.open(output_address);
			writeT << ratioResult;
			writeT.close();

			std::cout << imageAddresses[i] << "/ratio = " << ratioResult << std::endl;
		}
	}
}

//컬러디텍션 디텍션 결과 중, 디텍션 박스를 저장한다.
//이때 사용하는 욜로는 세이브 된 버전을 사용한다. ㅇㅇ
void RobustOptimalExperiment::SaveColorDetection_DetectionBox_txt(ColorDetection* colorDetection_Module, 
	const int& hueMargin, const int& medianFilterSize, const int& downSamplingSize)
{
	auto& colorDetection_Module_ = *colorDetection_Module;

	std::vector<std::string> imageFileNames;
	std::vector<std::string> testImageFileAdds = getFilenames("./testImages_resized", &imageFileNames);

	InBoxChecker inboxChecker_Tool;

	colorDetection_Module_.ClearColorDetectionRange();

	colorDetection_Module_.MakeColorDetection("autoColorParameter_1_B1_M1", hueMargin);
	colorDetection_Module_.MakeColorDetection("autoColorParameter_1_B2_M1", hueMargin);
	colorDetection_Module_.MakeColorDetection("autoColorParameter_1_B3_M1", hueMargin);

	colorDetection_Module_.MakeColorDetection("autoColorParameter_2_B1_M1", hueMargin);
	colorDetection_Module_.MakeColorDetection("autoColorParameter_2_B2_M1", hueMargin);
	colorDetection_Module_.MakeColorDetection("autoColorParameter_2_B3_M1", hueMargin);

	colorDetection_Module_.SetMedianBlurSize((ColorDetection::eMedianBlurSizes)medianFilterSize);
	colorDetection_Module_.SetDownSamplingSize((ColorDetection::eDownSamplingSizes)downSamplingSize);

	int count = testImageFileAdds.size();

	for (int i = 0; i < testImageFileAdds.size(); i++)
	{
		//인박스 체커 클리어
		inboxChecker_Tool.ClearBox();
		//욜로디텍션박스 불러옴
		getYOLOv3DetectionBox(testImageFileAdds[i], &inboxChecker_Tool);

		//파일명을 분리하여 보드컬러와 밝기 레벨 정보를 분해
		std::vector<std::string> split_filename = splitStringByChar(imageFileNames[i], '_');
		//1~2까지
		int board_color = std::atoi(split_filename[0].c_str());
		//1~3까지
		int brightness_level = (split_filename[3])[2] - '0';

		//위의 정보를 바탕으로 컬러레인지 인덱스를 구함
		int color_range_num = (board_color - 1) * 3 + (brightness_level - 1);

		//컬러디텍션에 통과시킬 이미지를 불러옴
		cv::Mat testimg = cv::imread(testImageFileAdds[i]);
		cv::Mat HSVedFrame;

		//해당 컬러레인지에 통과시킨 후, 결과를 구함
		colorDetection_Module_.PushThroughImage(testimg, &HSVedFrame, &inboxChecker_Tool, color_range_num);

		//결과 인박스 체커에서 컬러 디텍션 정보만 분리해옴
		std::vector<InBoxChecker::BoxInfo> boxes = inboxChecker_Tool.GetBoxes(InBoxChecker::ColorDetection);

		inboxChecker_Tool.DrawBoxes(&testimg, InBoxChecker::YOLOv3);
		inboxChecker_Tool.DrawBoxes(&testimg, InBoxChecker::ColorDetection);
		cv::imshow("detected Color boxes", testimg);
		cv::waitKey(1);

		//해당 박스 정보를 새롭게 저장함.
		//뒤에 붙는 말은 _ColorDetection_Detection_Results.txt로 정했다.
		std::ofstream writeT;
		std::string resultSaveAdd = testImageFileAdds[i].substr(0, testImageFileAdds[i].length() - 4) + "_ColorDetection_Detection_Results.txt";
		writeT.open(resultSaveAdd);
		for (int j = 0; j < boxes.size(); j++)
		{
			writeT << boxes[j].Name << "," << boxes[j].Box.x << "," << boxes[j].Box.y << "," << boxes[j].Box.width << "," << boxes[j].Box.height << "\n";
		}
		writeT.close();

		std::cout << boxes.size() << " boxes saved (" << i + 1 << "/" << count << ")" << "\n";
	}

}