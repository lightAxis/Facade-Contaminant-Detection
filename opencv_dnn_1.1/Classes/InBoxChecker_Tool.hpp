#pragma once

#include <vector>
#include <string>
#include <opencv2\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

//-------------------------------------------------------------
class InBoxChecker
{
public:
	enum eModuleType { YOLOv3 = 0, ColorDetection = 1, Grayscale = 2 };
	typedef struct BoxInfo
	{
		cv::Rect Box;
		eModuleType Type;
		std::string Name;
	};
private:
	std::vector<BoxInfo> mBoxes;
	bool mbCheckInBox;
public:
	InBoxChecker();
	void AddBox(const cv::Rect& addBox, const eModuleType& type, const std::string& name);
	void ClearBox();
	bool CheckInBox(const cv::Point& point, int* boxWidth);
	int GetBoxCount();
	BoxInfo GetBox(const int& index);
	std::vector<BoxInfo> GetBoxes(const eModuleType& moduleType);
	void DrawBoxes(cv::Mat* currFrame, const eModuleType& Type);
};