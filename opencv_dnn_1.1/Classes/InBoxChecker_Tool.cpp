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

//생성자
InBoxChecker::InBoxChecker()
{
	mbCheckInBox = false;
	mBoxes.clear();
}

//박스 추가하기
void InBoxChecker::AddBox(const cv::Rect& addBox, const eModuleType& type, const std::string& name )
{
	BoxInfo newInfo = BoxInfo();
	newInfo.Box = addBox;
	newInfo.Type = type;
	newInfo.Name = name;
	mBoxes.push_back(newInfo);
}

//박스 목록 클리어
void InBoxChecker::ClearBox()
{
	mBoxes.clear();
}

//박스 안에 포함되어있는지 검사
bool InBoxChecker::CheckInBox(const cv::Point& point, int* boxWidth)
{
	auto& boxWidth_ = *boxWidth;

	if (mBoxes.empty()) { return false; }

	for (int i = 0; i < mBoxes.size(); i++)
	{
		if (mBoxes[i].Box.contains(point))
		{
			boxWidth_ = mBoxes[i].Box.width;
			return true;
		}
	}

	return false;
}

//현재 박스가 몇개인지 개수를 가져오는 함수
int InBoxChecker::GetBoxCount()
{
	return mBoxes.size();
}

//index_번째의 박스 인포를 가져오는 함수
InBoxChecker::BoxInfo InBoxChecker::GetBox(const int& index)
{
	return mBoxes.at(index);
}

void InBoxChecker::DrawBoxes(cv::Mat* currFrame, const eModuleType& type)
{
	auto& currFrame_ = *currFrame;

	int size = mBoxes.size();

	cv::Scalar drawColor = cv::Scalar(255, 0, 0);

	switch (type)
	{
		case ColorDetection:
		{
			drawColor = cv::Scalar(0, 255, 0);
			break;
		}
	}

	for (int i = 0; i < size; i++)
	{
		if (mBoxes[i].Type == type)
		{
			cv::rectangle(currFrame_, mBoxes[i].Box, drawColor, 2);

			int baseLine;
			cv::Size labelSize = getTextSize(mBoxes[i].Name, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			int put_y = cv::max(mBoxes[i].Box.y, labelSize.height);
			cv::putText(currFrame_, mBoxes[i].Name, cv::Point(mBoxes[i].Box.x, put_y), cv::FONT_HERSHEY_SIMPLEX, 0.75, drawColor, 2);
		}
	}
}

std::vector<InBoxChecker::BoxInfo> InBoxChecker::GetBoxes(const eModuleType& moduleType)
{
	std::vector<InBoxChecker::BoxInfo> result;
	for (int i = 0; i < mBoxes.size(); i++)
	{
		if (mBoxes[i].Type == moduleType)
		{
			result.push_back(mBoxes[i]);
		}
	}
	return result;
}