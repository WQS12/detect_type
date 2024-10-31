#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <omp.h>
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/imgproc/imgproc_c.h"
#include<opencv2/imgproc/types_c.h>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <cuda_provider_factory.h>
// 命名空间
using namespace std;
using namespace cv;
using namespace Ort;

// 自定义配置结构
struct Configuration
{
	public: 
	float confThreshold; // Confidence threshold置信度阈值
	float nmsThreshold;  // Non-maximum suppression threshold非最大抑制阈值
	float objThreshold;  //Object Confidence threshold对象置信度阈值
	string modelpath;
};

// 定义BoxInfo结构类型
typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;

class YOLOv5
{
public:
	YOLOv5(Configuration config);
	void detect(Mat& frame);
private:
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;
	int num_classes;

    string classes[80] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus",
                            "train", "truck", "boat", "traffic light", "fire hydrant",
                            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                            "skis", "snowboard", "sports ball", "kite", "baseball bat",
                            "baseball glove", "skateboard", "surfboard", "tennis racket",
                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                            "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
                            "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
                            "sink", "refrigerator", "book", "clock", "vase", "scissors",
                            "teddy bear", "hair drier", "toothbrush"};


	const bool keep_ratio = true;
	vector<float> input_image_;		// 输入图片
	void normalize_(Mat img);		// 归一化函数
	void nms(vector<BoxInfo>& input_boxes);  
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1"); // 初始化环境
	Session *ort_session = nullptr;    // 初始化Session指针选项
	SessionOptions sessionOptions = SessionOptions();  //初始化Session对象
	//SessionOptions sessionOptions;
	vector<char*> input_names;  // 定义一个字符指针vector
	vector<char*> output_names; // 定义一个字符指针vector
	vector<vector<int64_t>> input_node_dims; // >=1 outputs  ，二维vector
	vector<vector<int64_t>> output_node_dims; // >=1 outputs ,int64_t C/C++标准
};

YOLOv5::YOLOv5(Configuration config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	this->num_classes = 80;//sizeof(this->classes)/sizeof(this->classes[0]);  // 类别数量
	this->inpHeight = 640;
	this->inpWidth = 640;
	
	string model_path = config.modelpath;


	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  //设置图优化类型

	ort_session = new Session(env, (const char*)model_path.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();  //输入输出节点数量                         
	size_t numOutputNodes = ort_session->GetOutputCount(); 
	AllocatorWithDefaultOptions allocator;   // 配置输入输出节点内存
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));		// 内存
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);   // 类型
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();  // 
		auto input_dims = input_tensor_info.GetShape();    // 输入shape
		input_node_dims.push_back(input_dims);	// 保存
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	this->nout = output_node_dims[0][2];      // 5+classes
	this->num_proposal = output_node_dims[0][1];  // pre_box

}

Mat YOLOv5::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)  //修改图片大小并填充边界防止失真
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);  //等比例缩小，防止失真
			*top = (int)(this->inpHeight - *newh) * 0.5;  //上部缺失部分
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114); //上部填补top大小，下部填补剩余部分，左右不填补
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLOv5::normalize_(Mat img)  //归一化
{

	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());  // vector大小

	for (int c = 0; c < 3; c++)  // bgr
	{
		for (int i = 0; i < row; i++)  // 行
		{
			for (int j = 0; j < col; j++)  // 列
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];  // Mat里的ptr函数访问任意一行像素的首地址,2-c:表示rgb
				this->input_image_[c * row * col + i * col + j] = pix / 255.0; //将每个像素块归一化后装进容器
			}
		}
	}
}

void YOLOv5::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列
	vector<bool> remove_flags(input_boxes.size(),false);
	auto iou = [](const BoxInfo& box1,const BoxInfo& box2)
	{
		float xx1 = max(box1.x1, box2.x1);
		float yy1 = max(box1.y1, box2.y1);
		float xx2 = min(box1.x2, box2.x2);
		float yy2 = min(box1.y2, box2.y2);
		// 交集
		float w = max(0.0f, xx2 - xx1 + 1);
		float h = max(0.0f, yy2 - yy1 + 1);
		float inter_area = w * h;
		// 并集
		float union_area = max(0.0f,box1.x2-box1.x1) * max(0.0f,box1.y2-box1.y1)
						   + max(0.0f,box2.x2-box2.x1) * max(0.0f,box2.y2-box2.y1) - inter_area;
		return inter_area / union_area;
	};
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if(remove_flags[i]) continue;
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if(remove_flags[j]) continue;
			if(input_boxes[i].label == input_boxes[j].label && iou(input_boxes[i],input_boxes[j])>=this->nmsThreshold)
			{
				remove_flags[j] = true;
			}
		}
	}
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &remove_flags](const BoxInfo& f) { return remove_flags[idx_t++]; }), input_boxes.end());
}


void YOLOv5::detect(Mat& frame)
{
    vector<BoxInfo> generate_boxes; // 在顶部声明，确保在整个函数中都可以访问该变量
    int newh = 0, neww = 0, padh = 0, padw = 0;
    Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw); // 调整图像大小并添加边界填充以防止形变
    this->normalize_(dstimg); // 对图像进行归一化处理

    // 定义输入矩阵
    array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); // 创建内存信息对象，用于分配CPU内存
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size()); // 创建输入张量

    // 运行推理
    vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, input_names.data(), &input_tensor_, 1, output_names.data(), output_names.size());

    float* pdata = ort_outputs[0].GetTensorMutableData<float>(); // 获取输出数据
    float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww; // 计算原图与新图的高度和宽度比

    for (int i = 0; i < num_proposal; ++i) 
    {
        int index = i * nout;
        float obj_conf = pdata[index + 4]; // 获取目标的置信度
        if (obj_conf > this->objThreshold) // 如果置信度高于阈值
        {
            int class_idx = 0;
            float max_class_socre = 0;
            for (int k = 0; k < this->num_classes; ++k)
            {
                if (pdata[k + index + 5] > max_class_socre) // 查找最高类别分数
                {
                    max_class_socre = pdata[k + index + 5];
                    class_idx = k;
                }
            }
            max_class_socre *= obj_conf; // 类别分数乘以置信度
            if (max_class_socre > this->confThreshold) // 如果最终得分高于置信度阈值
            {
                float cx = pdata[index];
                float cy = pdata[index + 1];
                float w = pdata[index + 2];
                float h = pdata[index + 3];
                float xmin = (cx - padw - 0.5 * w) * ratiow;
                float ymin = (cy - padh - 0.5 * h) * ratioh;
                float xmax = (cx - padw + 0.5 * w) * ratiow;
                float ymax = (cy - padh + 0.5 * h) * ratioh;

                generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx }); // 添加检测框信息
            }
        }
    }

    nms(generate_boxes); // 应用非最大抑制(NMS)去除重叠的检测框
    for (size_t i = 0; i < generate_boxes.size(); ++i)
    {
        int xmin = int(generate_boxes[i].x1);
        int ymin = int(generate_boxes[i].y1);
        rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2); // 在图像上画出检测框
        string label = format("%.2f", generate_boxes[i].score); // 格式化标签
        label = this->classes[generate_boxes[i].label] + ":" + label; // 添加类别名称到标签
        putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1); // 将标签绘制在检测框上方
    }
}


int main(int argc, char *argv[])
{
    if (argc < 2) {  // 确保至少有一个参数被传入
        cerr << "Usage: " << argv[0] << " <video_path>" << endl;  // 提示用户提供视频路径
        return -1;
    }

    VideoCapture cap(argv[1]);  // 尝试从文件路径打开视频
    if (!cap.isOpened()) {  // 检查视频是否成功打开
        cerr << "Failed to open video file: " << argv[1] << endl;
        return -1;
    }

    Configuration yolo_nets = { 0.3, 0.5, 0.3, "model.onnx" }; // 初始化属性
    YOLOv5 yolo_model(yolo_nets);

    Mat frame;
    namedWindow("Detection Output", WINDOW_NORMAL); // 创建窗口

    while (true) {
        cap >> frame; // 读取新的一帧
        if (frame.empty()) break; // 如果帧为空，则结束循环

        double t = (double)getTickCount(); // 获取开始时间点
        yolo_model.detect(frame);
        t = ((double)getTickCount() - t) / getTickFrequency(); // 计算推理耗时
        cout << "Detection Time: " << t << " seconds." << endl; // 打印推理时间

        imshow("Detection Output", frame);
        char key = (char) waitKey(1); // 等待1毫秒，检查是否有按键操作
        if (key == 27 || key == 'q' || key == 'Q') break; // 如果按下ESC或Q，则退出循环
    }

    return 0;
}
