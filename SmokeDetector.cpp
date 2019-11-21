

// SmokeDetector.cpp : Defines the entry point for the console application.
//


#include <torch/torch.h>
#include <iostream>
#include <torch/script.h> // One-stop header.
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cmath>




using namespace cv;
using namespace std;



void showImage(Mat image, String text)
{

	cv::putText(image, //target image
		text, //text
		cv::Point(35, 65), //top-left position
		cv::FONT_HERSHEY_DUPLEX,
		1.5,
		CV_RGB(255, 255, 0), //font color
		2);
	namedWindow("Display window");// Create a window for display.
	//resizeWindow("Display window", 600, 600);
	imshow("Display window", image);
	waitKey(1);
}




int main(int argc, const char* argv[]) {

	String label;


	// Deserialize the ScriptModule from a file using torch::jit::load().
	torch::jit::script::Module module = torch::jit::load("model_trace_vgg16.pt");
	cout << "OK\n";

	cv::VideoCapture capture("1# CT-CL-KMS-06@2019-11-08T200011.430Z.mkv");
	int start_frame_number = 4250;

	capture.set(CAP_PROP_POS_FRAMES, start_frame_number);
	cv::VideoWriter writer; 
	cv::Size ProcessSize(80,80);
	float UnitCoeff = 1.0f / 255.0f;

	while (capture.grab())
	{
		cv::Mat Frame;
		capture.retrieve(Frame);
		if (Frame.empty() == false)
		{
			cv::Mat ProcessBGRFrame, ProcessRGBFrame;
			cv::resize(Frame, ProcessBGRFrame, ProcessSize);
			cv::cvtColor(ProcessBGRFrame, ProcessRGBFrame, CV_BGR2RGB);
			// convert [unsigned int] to [float]
			ProcessRGBFrame.convertTo(ProcessRGBFrame, CV_32FC3, UnitCoeff);

			auto input_tensor = torch::from_blob(ProcessRGBFrame.data, { 1, ProcessRGBFrame.cols, ProcessRGBFrame.rows, ProcessRGBFrame.channels() });

			input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
			input_tensor[0][0] = input_tensor[0][0].sub_(0.485).div_(0.229);
			input_tensor[0][1] = input_tensor[0][1].sub_(0.456).div_(0.224);
			input_tensor[0][2] = input_tensor[0][2].sub_(0.406).div_(0.225);

			auto start = chrono::steady_clock::now();
			torch::Tensor out_tensor = module.forward({ input_tensor }).toTensor();
			auto end = chrono::steady_clock::now();
			auto diff = end - start;
			cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;

			auto results = out_tensor.sort(-1, true);
			auto softmax = std::get<0>(results)[0].softmax(0);
			auto index = std::get<1>(results)[0];
			std::cout << "Predicted class : " << index[0] << std::endl;
			std::cout << out_tensor << std::endl;

			if ((index[0].item<int64_t>()) == 0)
			{
				label = "Fire";
			}
			if ((index[0].item<int64_t>()) == 1)
			{
				label = "Neutral";
			}
			if ((index[0].item<int64_t>()) == 2)
			{
				label = "Smoke";
			}

			showImage(Frame, label);
			if(writer.isOpened() == false)
				writer.open("Debug15.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Frame.size() );
			writer.write(Frame);
		}
		//std::cout << "OK";
	}
}
