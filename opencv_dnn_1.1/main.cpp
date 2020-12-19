#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\dnn.hpp>
#include <opencv2\dnn/shape_utils.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <sstream>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>
#include <algorithm> 
#include <queue>
//#include <Windows.h>
#include <cmath>

#include <libCamCap.h>

#include "Classes/OCam_Input.hpp"
#include "Classes/PylonCameraModule.hpp"

#include "Classes/InBoxChecker_Tool.hpp"

#include "Classes/GrayScale_Module.hpp"
#include "Classes/ColorDetection_Module.hpp"
#include "Classes/YOLO_v3_Module.hpp"

#include "Classes/RobustOptimalExperiment_Tool.hpp"


using namespace std;


//---------------------------------------------------------
static const int IMAGE_WIDTH = 640;     // pixel
static const int IMAGE_HEIGHT = 480;    // pixel
static const double IMAGE_FPS = 60;   // frame per second

static const long EXPOSURE = -5;
static const long GAIN = 128;
static const long WB_BLUE = 216;
static const long WB_RED = 121;


//-------------------------------------------------------------
float confThreshold = 0.1; // Confidence threshold
float nmsThreshold = 0.5;  // Non-maximum suppression threshold
int inpWidth = 608;        // Width of network's input image
int inpHeight = 608;

//--------------------------------------------------------------
int pos_hue = 20;
int mid_hue = 10;
int neg_hue = 0;

//--------------------------------------------------------------

//미리 쓸 Frame들 장착
cv::Mat frame;
cv::Mat detectedFrame;
cv::Mat HLS_frame;

//미리 쓸 BoxChecker 선언
InBoxChecker InBoxChecker_Tool = InBoxChecker();

//dnn network settings for object detection
YOLO_v3_DNN YOLO_v3_Module = YOLO_v3_DNN(confThreshold, nmsThreshold, inpWidth, inpHeight);

//미리 쓸 ColorDetection 모듈 선언
ColorDetection ColorDetection_Module = ColorDetection(ColorDetection::F9x9, ColorDetection::S11x11);

//미리 쓸 GrayScale 모듈 선언
GrayScaleCalculator GrayScale_Module = GrayScaleCalculator(GrayScaleCalculator::RGB_Luminance);

//오캠 카메라 임포트
//OCam OCam_Camera = OCam(OCam::W640_H480, OCam::FPS60);
//테스트용 웹캠 임포트
//cv::VideoCapture cap(0);
//Basler 카메라 pylon sdk로 임포트
PylonCamera PylonCam_Module;

//--------------------------------디버그용 함수들, 클릭 이벤트 등등--------------------------------

cv::Mat test_HSV;
cv::Mat test_gray;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		cout << "pos : [" << x << "," << y << "]" << endl;
		cout << "original HSL Value : " << (int)(test_HSV.at<cv::Vec3b>(cv::Point(x, y))[0]) << "/" << (int)(test_HSV.at<cv::Vec3b>(cv::Point(x, y))[1]) << "/" << (int)(test_HSV.at<cv::Vec3b>(cv::Point(x, y))[2]) << endl;
	}
}

void CallBackTestFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		cout << "pos : [" << x / 2 << "," << y / 2 << "]" << endl;
		//cout << "Value : " << (int)(test_HSV.at<cv::Vec3b>(cv::Point(x, y))[0]) << "/" << (int)(test_HSV.at<cv::Vec3b>(cv::Point(x, y))[1]) << "/" << (int)(test_HSV.at<cv::Vec3b>(cv::Point(x, y))[2]) << endl;
	}
}


void calcHistAndDislay(cv::Mat* testImg);
void showHuesAndsats(cv::Mat* testImage);
void get3GrayImages(cv::Mat* img);

//--------------------------------------------------------------
int main()
{
	//오캠 카메라 세팅
	//OCam_Camera.setCameraParameters(EXPOSURE, GAIN, WB_BLUE, WB_RED);
	//OCam_Camera.StartOCam();

	//베슬러 카메라 세팅
	PylonCam_Module.PrepareCamera();


	//욜로모듈 세팅
	string folderName = "./birdPoop";
	string classesFile = folderName + "/" + "object.names";
	string modelConfiguration = folderName + "/" + "yolov3-obj-custom.cfg";
	string modelWeights = folderName + "/" + "yolov3-obj-custom_1000.weights";
	YOLO_v3_Module.MakeYOLONetFromFile(classesFile, modelConfiguration, modelWeights);


	//컬러 디텍션 모듈 세팅
	ColorDetection_Module.AddColorDetectionRange(pos_hue, mid_hue, neg_hue,"Rust", 130);
	ColorDetection_Module.MakeColorDetection("autoColorParameter_2_B2_M1", 10);


	//메인 루프문 시작
	while (true)
	{
		//오캠에서 프레임 가져오기
		//frame = OCam_Camera.getFrame();
		//웹캠에서 프레임 가져오기
		//cap.read(frame);
		//파일에서 프레임 가져오기
		//if (change_test_invorinment(frame) == false)
		//{
		//	break;
		//}

		//탐색 결과가 그려질 다른 프레임 지정
		frame.copyTo(detectedFrame);

		if (PylonCam_Module.GrabCameraFrame(&frame))
		{
			cv::imshow("pylon original frame", frame);
			if (cv::waitKey(1) == 27)
			{
				break;
			}
		}
		else
		{
			break;
		}

		//-----욜로 모듈 부분

		//욜로에 보내서 프레임 분석하고 박스값 따로 뽑은 후, 이미지 처리 진행
		YOLO_v3_Module.PassThrough(&frame);
		YOLO_v3_Module.GetObjectRects(&InBoxChecker_Tool);
		YOLO_v3_Module.DrawBoxes(&detectedFrame, true);

		cv::imshow("YOLO Result", detectedFrame);
		cv::imwrite("./imwrite_imgs/YOLO_Result.jpg", detectedFrame);
		if (cv::waitKey(0) == 27);

		//욜로 통과 결과
		cv::imshow("YOLO v3 result", detectedFrame);

		//-----컬러 디텍션 모듈 부분

		//컬러 디텍션 모듈에 보냄
		int detected_pixels = ColorDetection_Module.PushThroughImage(frame, &test_HSV, &InBoxChecker_Tool);
		float Detected_pixel_ratio = ((float)detected_pixels) / (frame.cols * frame.rows);
		cout << "detected Color Area Ratio : " << Detected_pixel_ratio << " / Pixels : " << detected_pixels << endl;


		//컬러 디텍션 모듈 통과 결과에다가 박스 마저 침
		InBoxChecker_Tool.DrawBoxes(&detectedFrame, InBoxChecker::ColorDetection);

		//컬러 디텍션까지 통과한 결과
		cv::imshow("colorDetectedWithMedianBlur Result", detectedFrame);
		cv::imwrite("./imwrite_imgs/colorDetectedWithMedianBlur_Result.jpg", detectedFrame);
		if (cv::waitKey(0) == 27);

		//break;
		cv::Mat back_sample = cv::imread("./autoColorParameters/autoColorParameter_fig_test/Rust/Background/바탕 피규어 예시 1.png");
		cv::Mat color_sample = cv::imread("./autoColorParameters/autoColorParameter_fig_test/Rust/ColorArea/컬러 피규어 예시 1.png");

		calcHistAndDislay(&back_sample);
		calcHistAndDislay(&color_sample);

		//그레이스케일 모듈에 나머지 영역을 통과시켜서 평균 밝기 계산
		float res = GrayScale_Module.CalcGrayScale(frame, &InBoxChecker_Tool);
		cout << "cur grayscale : " << res << endl;

		//마우스 콜백 함수 설정
		cv::setMouseCallback("original", CallBackFunc, NULL);
		cv::setMouseCallback("colorDetectedWithMedianBlur Result", CallBackFunc, NULL);

		if (cv::waitKey(0) == 27)

			//press esc to escape
			if (cv::waitKey(1) == 27) break;

		InBoxChecker_Tool.ClearBox();


	}

	//오캠 스트리밍 닫기
	//OCam_Camera.CloseOCam();

	//베슬러 카메라 닫기
	PylonCam_Module.CloseCamera();

	return 0;
}

void calcHistAndDislay(cv::Mat* testImg)
{
	auto& testImg_ = *testImg;

	cv::imshow("testimg", testImg_);
	cv::Mat saturationHSV;
	cv::cvtColor(testImg_, saturationHSV, cv::COLOR_BGR2HSV);
	cv::MatND hist;

	const int channel_numbers[] = { 1 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges[1] = { channel_range };
	int number_bins = 255;

	int hist_w = 255 * 4;
	int hist_h = 255;

	int medianBlurSize = 9;


	cv::calcHist(&saturationHSV, 1, channel_numbers, cv::Mat(), hist, 1, &number_bins, channel_ranges);

	int bin_w = cvRound((double)hist_w / number_bins);

	cv::Mat histImage = cv::Mat(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < number_bins; i++)
	{
		cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(100, 100, 255), 2, 8, 0);
	}

	cv::namedWindow("histimage");
	cv::setMouseCallback("histimage", CallBackTestFunc, NULL);
	cv::imshow("histimage", histImage);


	cv::Mat medianBluredHSVImg;
	cv::medianBlur(saturationHSV, medianBluredHSVImg, medianBlurSize);
	cv::calcHist(&medianBluredHSVImg, 1, channel_numbers, cv::Mat(), hist, 1, &number_bins, channel_ranges);
	cv::Mat BluredhistImage = cv::Mat(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::normalize(hist, hist, 0, BluredhistImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < number_bins; i++)
	{
		cv::line(BluredhistImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(100, 100, 255), 2, 8, 0);
	}
	cv::namedWindow("histimage_Blured");
	cv::setMouseCallback("histimage_Blured", CallBackTestFunc, NULL);
	cv::imshow("histimage_Blured", BluredhistImage);

	const int hue_channel_numbers[] = { 0 };

	cv::calcHist(&medianBluredHSVImg, 1, hue_channel_numbers, cv::Mat(), hist, 1, &number_bins, channel_ranges);
	cv::Mat BluredhistImage_hue = cv::Mat(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::normalize(hist, hist, 0, BluredhistImage_hue.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < number_bins; i++)
	{
		cv::line(BluredhistImage_hue, cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(100, 100, 255), 2, 8, 0);
	}
	cv::namedWindow("histimage_Blured_hue");
	cv::setMouseCallback("histimage_Blured_hue", CallBackTestFunc, NULL);
	cv::imshow("histimage_Blured_hue", BluredhistImage_hue);


	cv::Mat splited_original[3];
	cv::split(saturationHSV, splited_original);
	cv::imshow("testImgSats", splited_original[1]);

	cv::Mat splited_blured[3];
	cv::split(medianBluredHSVImg, splited_blured);
	cv::imshow("testBluredSats", splited_blured[1]);

	cv::waitKey(0);
}

void showHuesAndsats(cv::Mat* testImage)
{
	auto& testImage_ = *testImage;
	cv::Mat hsved;
	cv::cvtColor(testImage_, hsved, cv::COLOR_BGR2HSV);
	cv::Mat hued = cv::Mat(testImage_.rows, testImage_.cols, CV_8UC3);
	cv::Mat hued_bgr;
	cv::Mat sated = cv::Mat(testImage_.rows, testImage_.cols, CV_8UC1);

	for (int r = 0; r < testImage_.rows; r++)
	{
		for (int c = 0; c < testImage_.cols; c++)
		{
			hued.at<cv::Vec3b>(r, c) = cv::Vec3b(hsved.at<cv::Vec3b>(r, c)[0], 0xFF, 127);
			sated.at<uchar>(r, c) = hsved.at<cv::Vec3b>(r, c)[1];
		}
	}

	cv::cvtColor(hued, hued_bgr, cv::COLOR_HSV2BGR);
	cv::imshow("original", testImage_);
	cv::imwrite("./imwrite_imgs/original_.jpg", testImage_);
	cv::imshow("hue map", hued_bgr);
	cv::imwrite("./imwrite_imgs/hue_map.jpg", hued_bgr);
	cv::imshow("sat map", sated);
	cv::imwrite("./imwrite_imgs/sat_map.jpg", sated);

	cv::waitKey(0);
}


void get3GrayImages(cv::Mat* img)
{
	auto& img_ = *img;

	cv::Mat YUV;
	cv::Mat YUV_c[3];
	cv::Mat YUV_Y;
	cv::Mat HSV;
	cv::Mat HSV_c[3];
	cv::Mat HSV_V;
	cv::Mat Lab;
	cv::Mat Lab_c[3];
	cv::Mat Lab_L;

	cv::cvtColor(img_, YUV, cv::COLOR_BGR2YUV);
	cv::cvtColor(img_, HSV, cv::COLOR_BGR2HSV);
	cv::cvtColor(img_, Lab, cv::COLOR_BGR2Lab);

	cv::split(YUV, YUV_c);
	cv::split(HSV, HSV_c);
	cv::split(Lab, Lab_c);

	cv::imshow("original", img_);
	cv::imshow("YUV", YUV_c[0]);
	cv::imshow("HSV", HSV_c[2]);
	cv::imshow("Lab", Lab_c[0]);

	cv::waitKey(0);

	cv::cvtColor(YUV_c[0], YUV_Y, cv::COLOR_GRAY2BGR);
	cv::cvtColor(HSV_c[2], HSV_V, cv::COLOR_GRAY2BGR);
	cv::cvtColor(Lab_c[0], Lab_L, cv::COLOR_GRAY2BGR);

	cv::imwrite("./imwrite_imgs/original.jpg", img_);
	cv::imwrite("./imwrite_imgs/YUVed img.jpg", YUV_Y);
	cv::imwrite("./imwrite_imgs/HSVed img.jpg", HSV_V);
	cv::imwrite("./imwrite_imgs/Labed img.jpg", Lab_L);
}
