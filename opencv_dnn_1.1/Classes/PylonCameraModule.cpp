

#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pylon/PylonIncludes.h>
#ifdef PYLON_WIN_BUILD
#include <pylon/PylonGUI.h>
#endif


class PylonCamera
{
private:
	int mSaveImages = 0; // define is images are to be saved, 0-no 1-yes
	int mRecordVideo = 0; // define is videos are to be saved, 0-no, 1-yes
	const int mCountOfImagesToGrab = -1; // number of images to be grabbed


	Pylon::CInstantCamera* mCamera; // pylon camera object
	Pylon::CImageFormatConverter mFormatConverter; // pylon formatConverter
	Pylon::CPylonImage mPylonImage; // pylon image format data

	cv::Size mFrameSize; // opencv frame size of pylon camera

public:
	void PrepareCamera(); // prepare pylon basler camera
	bool GrabCameraFrame(cv::Mat* currFrame); // grab the one frame and return openCV image
	void CloseCamera();

	//void test();
};

void PylonCamera::PrepareCamera()
{
	Pylon::PylonInitialize();

	try
	{
		//create an instant camera object with the camera device found first
		mCamera = new Pylon::CInstantCamera(Pylon::CTlFactory::GetInstance().CreateFirstDevice());

		// print the model name of the camera
		std::cout << "Using device" << mCamera->GetDeviceInfo().GetModelName() << std::endl;

		// get a camera nodemap in order to access camera parameters
		GenApi::INodeMap& nodemap = mCamera->GetNodeMap();
		// Load the User Set 1 user set
		//GenApi::CEnumerationPtr(nodemap.GetNode("UserSetSelector"))->FromString("UserSet1");
		//GenApi::CCommandPtr(nodemap.GetNode("UserSetLoad"))->Execute();

		// create pointers to access the camera width and height parameters
		GenApi::CIntegerPtr width = nodemap.GetNode("Width");
		GenApi::CIntegerPtr height = nodemap.GetNode("Height");

		//std::cout << GenApi::IsReadable(width) << std::endl;
		// define the image frame size
		//frameSize = cv::Size((int)width->GetValue(), (int)height->GetValue());

		// the parameter MaxNumBuffer can be used to control the count of buffers
		// allocated for grabbing. the default value of this parameter is 10
		mCamera->MaxNumBuffer = 5;

		// specify the output pixel format
		mFormatConverter.OutputPixelFormat = Pylon::PixelType_BGR8packed;

		//camera->StartGrabbing(c_countOfImagesToGrab, Pylon::GrabStrategy_LatestImageOnly);

	}
	catch (const Pylon::GenericException& e)
	{
		// Error handling.
		std::cout << "An exception occurred." << "\n" << e.GetDescription() << std::endl;
	}
}


bool PylonCamera::GrabCameraFrame(cv::Mat* currFrame)
{
	auto& currFrame_ = *currFrame;
	try
	{
		///*
		Pylon::CGrabResultPtr ptrGrabResult;
		// grab one image from camera
		mCamera->GrabOne(100, ptrGrabResult, Pylon::TimeoutHandling_ThrowException);

		// convert the grabbed buffer to a pylon image
		mFormatConverter.Convert(mPylonImage, ptrGrabResult);

		// create an OpenCV image from a pylon image
		if (ptrGrabResult->GrabSucceeded())
		{
			currFrame_ = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t*)mPylonImage.GetBuffer());
			return true;
		}
		return false;
		//*/


		/*
		if (camera->IsGrabbing())
		{
			Pylon::CGrabResultPtr ptrGrabResult;
			camera->RetrieveResult(100, ptrGrabResult, Pylon::TimeoutHandling_ThrowException);
			formatConverter.Convert(pylonImage, ptrGrabResult);
			frame = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t*)pylonImage.GetBuffer());
			return true;
		}
		*/

		return false;
	}
	catch (const Pylon::GenericException& e)
	{
		// Error handling.
		std::cout << "An exception occurred." << "\n" << e.GetDescription() << std::endl;

		return false;
	}
}

void PylonCamera::CloseCamera()
{
	//camera->StopGrabbing();
	Pylon::PylonTerminate();
}

/*
void PylonCamera::test()
{
	Pylon::PylonAutoInitTerm autoInitTerm;

	try {
		//create an instant camera object with the camera device found first
		Pylon::CInstantCamera camera(Pylon::CTlFactory::GetInstance().CreateFirstDevice());

		// print the model name of the camera
		std::cout << "Using device" << camera.GetDeviceInfo().GetModelName() << std::endl;

		// get a camera nodemap in order to access camera parameters
		GenApi::INodeMap& nodemap = camera.GetNodeMap();

		// create pointers to access the camera width and height parameters
		GenApi::CIntegerPtr width = nodemap.GetNode("Width");
		GenApi::CIntegerPtr height = nodemap.GetNode("Height");

		// the parameter MaxNumBuffer can be used to control the count of buffers
		// allocated for grabbing. the default value of this parameter is 10
		camera.MaxNumBuffer = 5;

		// create a pylon ImageFormatConverter object
		Pylon::CImageFormatConverter formatConverter;
		// specify the output pixel format
		formatConverter.OutputPixelFormat = Pylon::PixelType_BGR8packed;
		// create a PylonImage that will be used to create the OpenCV images later
		Pylon::CPylonImage pylonImage;
		// declare an integer variable to count the number of grabbed images
		// and create image file names with ascending number
		int grabbedImages = 0;

		// create an OpenCV image
		cv::Mat openCVImage;

		// define the image frame size
		cv::Size frameSize = cv::Size((int)width->GetValue(), (int)height->GetValue());

		// sets up free-running continuous acquisition
		camera.StartGrabbing(c_countOfImagesToGrab, Pylon::GrabStrategy_LatestImageOnly);

		// this smart pointer will receive the grab result data
		Pylon::CGrabResultPtr ptrGrabResult;

		//Camera.StopGrabbing() is called automatically by the RetreveResult() method
		// when c_countOfImagesToGrab images have been retrieved

		while (camera.IsGrabbing())
		{
			// wait for an image and then retrieve it. a timeout of 5000ms is used
			camera.RetrieveResult(5000, ptrGrabResult, Pylon::TimeoutHandling_ThrowException);

			// image grabbed successfully?
			if (ptrGrabResult->GrabSucceeded())
			{
				//access the image data
				std::cout << "Size X: " << ptrGrabResult->GetWidth() << std::endl;
				std::cout << "Size Y: " << ptrGrabResult->GetHeight() << std::endl;

				// convert the grabbed buffer to a pylon image
				formatConverter.Convert(pylonImage, ptrGrabResult);

				// create an OpenCV image from a pylon image
				openCVImage = cv::Mat(ptrGrabResult->GetHeight(), ptrGrabResult->GetWidth(), CV_8UC3, (uint8_t*)pylonImage.GetBuffer());

				// display the OpenCV image to window
				cv::namedWindow("pylon camera input to opencv Test");
				cv::imshow("pylon camera input to opencv Test", openCVImage);
				// wait for 1 sec
				cv::waitKey(0);



			}
		}




	}
	catch (const Pylon::GenericException& e)
	{
		// Error handling.
		std::cout << "An exception occurred." << "\n" << e.GetDescription() << std::endl;

	}

	camera.StopGrabbing();
	Pylon::PylonTerminate();


}
*/