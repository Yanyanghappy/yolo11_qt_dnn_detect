#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "AclLiteUtils.h"
#include "AclLiteImageProc.h"
#include "AclLiteResource.h"
#include "AclLiteError.h"
#include "AclLiteModel.h"
#include "label.h"
#include <chrono>


using namespace std;
using namespace cv;
typedef enum Result
{
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef struct BoundBox
{
    float x;
    float y;
    float width;
    float height;
    float score;
    size_t classIndex;
    size_t index;
} BoundBox;

bool sortScore(BoundBox box1, BoundBox box2)
{
    return box1.score > box2.score;
}

class SampleYOLOV11
{
public:
    SampleYOLOV11(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight);
    Result InitResource();
    Result ProcessInput(cv::Mat &iImg);
    Result Inference(std::vector<InferenceOutput> &inferOutputs);
    Result GetResult(std::vector<InferenceOutput> &inferOutputs, cv::Mat &iImg, size_t imageIndex, bool release);
    ~SampleYOLOV11();

private:
    void ReleaseResource();
    AclLiteResource aclResource_;
    AclLiteImageProc imageProcess_;
    AclLiteModel model_;
    aclrtRunMode runMode_;
    ImageData resizedImage_;
    const char *modelPath_;
    int32_t modelWidth_;
    int32_t modelHeight_;
};

SampleYOLOV11::SampleYOLOV11(const char *modelPath, const int32_t modelWidth, const int32_t modelHeight) : modelPath_(modelPath), modelWidth_(modelWidth), modelHeight_(modelHeight)
{
}

SampleYOLOV11::~SampleYOLOV11()
{
    ReleaseResource();
}

Result SampleYOLOV11::InitResource()
{
    // init acl resource
    AclLiteError ret = aclResource_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("resource init failed, errorCode is %d", ret);
        return FAILED;
    }

    ret = aclrtGetRunMode(&runMode_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("get runMode failed, errorCode is %d", ret);
        return FAILED;
    }

    // init dvpp resource
    ret = imageProcess_.Init();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("imageProcess init failed, errorCode is %d", ret);
        return FAILED;
    }

    // load model from file
    ret = model_.Init(modelPath_);
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("model init failed, errorCode is %d", ret);
        return FAILED;
    }
     
    return SUCCESS;
}

struct ImageDataFloat {
    acldvppPixelFormat format;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t alignWidth = 0;
    uint32_t alignHeight = 0;
    uint32_t size = 0;
    std::shared_ptr<float> data = nullptr;
};

Result SampleYOLOV11::ProcessInput(cv::Mat &iImg)
{
    std::vector<int> iImgSize={640,640};
    int m_imgWidth=iImg.cols;
    int m_imgHeight=iImg.rows;
    cv::Mat oImg = iImg.clone();
    cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    cv::Mat detect_image;
    cv::resize(oImg, detect_image, cv::Size(iImgSize.at(0), iImgSize.at(1)));
    float* imageBytes;
    
    // data standardization
    float meanRgb[3] = {0, 0, 0};
    float stdRgb[3]  = {1/255.0f, 1/255.0f, 1/255.0f};
    int32_t channel = detect_image.channels();
    int32_t resizeHeight = detect_image.rows;
    int32_t resizeWeight = detect_image.cols;
    imageBytes = (float *) malloc(channel * (iImgSize.at(0)) * (iImgSize.at(1)) * sizeof(float));
    memset(imageBytes, 0, channel * (iImgSize.at(0)) * (iImgSize.at(1)) * sizeof(float));
    // image to bytes with shape HWC to CHW, and switch channel BGR to RGB
    for (int c = 0; c < channel; ++c) {
        for (int h = 0; h < resizeHeight; ++h) {
            for (int w = 0; w < resizeWeight; ++w) {
                int dstIdx = c * resizeHeight * resizeWeight + h * resizeWeight + w;
                imageBytes[dstIdx] = static_cast<float>(
                        (detect_image.at<cv::Vec3b>(h, w)[c] -
                         1.0f * meanRgb[c]) * 1.0f * stdRgb[c] );
            }
        }
    }
    AclLiteError ret = model_.CreateInput(static_cast<void *>(imageBytes),
                             channel * iImgSize.at(0) * iImgSize.at(1) * sizeof(float));
    if (ret == FAILED) {
        ACLLITE_LOG_ERROR("CreateInput failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV11::Inference(std::vector<InferenceOutput> &inferOutputs)
{
    // inference
    AclLiteError ret = model_.Execute(inferOutputs);
    if (ret != ACL_SUCCESS)
    {
        ACLLITE_LOG_ERROR("execute model failed, errorCode is %d", ret);
        return FAILED;
    }
    return SUCCESS;
}

Result SampleYOLOV11::GetResult(std::vector<InferenceOutput> &inferOutputs,
                                cv::Mat &srcImage, size_t imageIndex, bool release)
{
    uint32_t outputDataBufId = 0;
    float *classBuff = static_cast<float *>(inferOutputs[outputDataBufId].data.get());
    // confidence threshold
    float confidenceThreshold = 0.35;

    // class number
    size_t classNum = 80;

    //// number of (x, y, width, hight)
    size_t offset = 4;

    // total number of boxs yoloV11 [1,84,8400]
    size_t modelOutputBoxNum = 8400; 

    // read source image from file
    int srcWidth = srcImage.cols;
    int srcHeight = srcImage.rows;

    // filter boxes by confidence threshold
    vector<BoundBox> boxes;
    size_t yIndex = 1;
    size_t widthIndex = 2;
    size_t heightIndex = 3;

    // size_t all_num = 1 * 84 * 8400 ; // 705,600

    for (size_t i = 0; i < modelOutputBoxNum; ++i)
    {

        float maxValue = 0;
        size_t maxIndex = 0;
        for (size_t j = 0; j < classNum; ++j)
        {

            float value = classBuff[(offset + j) * modelOutputBoxNum + i];
            if (value > maxValue)
            {
                // index of class
                maxIndex = j;
                maxValue = value;
            }
        }

        if (maxValue > confidenceThreshold)
        {
            BoundBox box;
            box.x = classBuff[i] * srcWidth / modelWidth_;
            box.y = classBuff[yIndex * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.width = classBuff[widthIndex * modelOutputBoxNum + i] * srcWidth / modelWidth_;
            box.height = classBuff[heightIndex * modelOutputBoxNum + i] * srcHeight / modelHeight_;
            box.score = maxValue;
            box.classIndex = maxIndex;
            box.index = i;
            if (maxIndex < classNum)
            {
                boxes.push_back(box);
            }
        }
    }

    ACLLITE_LOG_INFO("filter boxes by confidence threshold > %f success, boxes size is %ld", confidenceThreshold,boxes.size());

    // filter boxes by NMS
    vector<BoundBox> result;
    result.clear();
    float NMSThreshold = 0.45;
    int32_t maxLength = modelWidth_ > modelHeight_ ? modelWidth_ : modelHeight_;
    std::sort(boxes.begin(), boxes.end(), sortScore);
    BoundBox boxMax;
    BoundBox boxCompare;
    while (boxes.size() != 0)
    {
        size_t index = 1;
        result.push_back(boxes[0]);
        while (boxes.size() > index)
        {
            boxMax.score = boxes[0].score;
            boxMax.classIndex = boxes[0].classIndex;
            boxMax.index = boxes[0].index;

            // translate point by maxLength * boxes[0].classIndex to
            // avoid bumping into two boxes of different classes
            boxMax.x = boxes[0].x + maxLength * boxes[0].classIndex;
            boxMax.y = boxes[0].y + maxLength * boxes[0].classIndex;
            boxMax.width = boxes[0].width;
            boxMax.height = boxes[0].height;

            boxCompare.score = boxes[index].score;
            boxCompare.classIndex = boxes[index].classIndex;
            boxCompare.index = boxes[index].index;

            // translate point by maxLength * boxes[0].classIndex to
            // avoid bumping into two boxes of different classes
            boxCompare.x = boxes[index].x + boxes[index].classIndex * maxLength;
            boxCompare.y = boxes[index].y + boxes[index].classIndex * maxLength;
            boxCompare.width = boxes[index].width;
            boxCompare.height = boxes[index].height;

            // the overlapping part of the two boxes
            float xLeft = max(boxMax.x, boxCompare.x);
            float yTop = max(boxMax.y, boxCompare.y);
            float xRight = min(boxMax.x + boxMax.width, boxCompare.x + boxCompare.width);
            float yBottom = min(boxMax.y + boxMax.height, boxCompare.y + boxCompare.height);
            float width = max(0.0f, xRight - xLeft);
            float hight = max(0.0f, yBottom - yTop);
            float area = width * hight;
            float iou = area / (boxMax.width * boxMax.height + boxCompare.width * boxCompare.height - area);

            // filter boxes by NMS threshold
            if (iou > NMSThreshold)
            {
                boxes.erase(boxes.begin() + index);
                continue;
            }
            ++index;
        }
        boxes.erase(boxes.begin());
    }

    ACLLITE_LOG_INFO("filter boxes by NMS threshold > %f success, result size is %ld", NMSThreshold,result.size());
 
    // opencv draw label params
    const double fountScale = 0.5;
    const uint32_t lineSolid = 2;
    const uint32_t labelOffset = 11;
    const cv::Scalar fountColor(0, 0, 255); // BGR
    const vector<cv::Scalar> colors{
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0),
        cv::Scalar(0, 0, 255)};

    int half = 2;
    for (size_t i = 0; i < result.size(); ++i)
    {
        cv::Point leftUpPoint, rightBottomPoint;
        leftUpPoint.x = result[i].x - result[i].width / half;
        leftUpPoint.y = result[i].y - result[i].height / half;
        rightBottomPoint.x = result[i].x + result[i].width / half;
        rightBottomPoint.y = result[i].y + result[i].height / half;
        cv::rectangle(srcImage, leftUpPoint, rightBottomPoint, colors[i % colors.size()], lineSolid);
        string className = label[result[i].classIndex];
        string markString = to_string(result[i].score) + ":" + className;

        ACLLITE_LOG_INFO("object detect [%s] success", markString.c_str());

        cv::putText(srcImage, markString, cv::Point(leftUpPoint.x, leftUpPoint.y + labelOffset),
                    cv::FONT_HERSHEY_COMPLEX, fountScale, fountColor);
    }
    string savePath = "../imgs/out_" + to_string(imageIndex) + ".jpg";
    // 显示图片
    cv::imshow("Display Window", srcImage);
    // 等待30ms，并检查是否按下了ESC键
    if (cv::waitKey(10) == 27) { // ESC键的ASCII码是27
        std::cout << "ESC pressed - exiting..." << std::endl;
    }
    // if (release)
    // {
        // free(classBuff);
        // classBuff = nullptr;
    //}
    return SUCCESS;
}

void SampleYOLOV11::ReleaseResource()
{
    model_.DestroyResource();
    imageProcess_.DestroyResource();
    aclResource_.Release();
}

int main()
{
    const char *modelPath = "../model/yolov11m-face.om";
    const string imagePath = "../data";
    const int32_t modelWidth = 640;
    const int32_t modelHeight = 640;
    SampleYOLOV11 sampleYOLO(modelPath, modelWidth, modelHeight);
    Result ret = sampleYOLO.InitResource();
    if (ret == FAILED)
    {
        ACLLITE_LOG_ERROR("InitResource failed, errorCode is %d", ret);
        return FAILED;
    }
    bool release = false;
    // 创建窗口显示图片
    cv::namedWindow("Display Window", cv::WINDOW_AUTOSIZE);
        
    // Initialize the camera
    cv::VideoCapture camera(0, cv::CAP_V4L2); // Use 0 for the default camera
    if (!camera.isOpened()) {
        std::cerr << "Cannot open the webcam!" << std::endl;
        return -1;
    }

    // Set the codec to MJPG if it is supported
    bool success = camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    if (!success) {
        std::cerr << "Failed to set MJPG codec!" << std::endl;
    }

    // Optionally set the frame width and height
    // camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280.0);
    // camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720.0);

    // Get the width and height of the frames
    int frame_width = static_cast<int>(camera.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(camera.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame width: " << frame_width << ", Frame height: " << frame_height << std::endl;

    // Main loop to capture and display frames
    cv::Mat frame;
    int i = 0;
    while (true) {
        std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
        std::vector<InferenceOutput> inferOutputs;
        camera >> frame; // Capture a frame
        if (frame.empty()) {
            std::cerr << "Failed to capture frame!" << std::endl;
            break;
        }

        ret = sampleYOLO.ProcessInput(frame);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("ProcessInput image failed, errorCode is %d", ret);
            return FAILED;
        }
        // inference
        ret = sampleYOLO.Inference(inferOutputs);
        if (ret == FAILED)
        {
            ACLLITE_LOG_ERROR("Inference failed, errorCode is %d", ret);
            return FAILED;
        }

        //处理一下最后一张图片释放空间的情况release=true
        ret = sampleYOLO.GetResult(inferOutputs, frame, i, release);
        if (ret == FAILED)
        {
        
            ACLLITE_LOG_ERROR("GetResult failed, errorCode is %d", ret);
            return FAILED;
        }
        i++;
        std::chrono::time_point<std::chrono::steady_clock> end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        ACLLITE_LOG_INFO("Inference elapsed time : %f s , fps is %f", elapsed.count(), 1 / elapsed.count());
    }
    camera.release();
    cv::destroyAllWindows();
    return SUCCESS;
}
