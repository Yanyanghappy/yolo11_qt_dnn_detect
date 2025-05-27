#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QImage>
#include <QString>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <atomic>
#include "safequeue.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void updateFrame();
    void on_startButton_clicked();
    void on_stopButton_clicked();

private:
    Ui::MainWindow *ui;
    QTimer *timer;
    cv::VideoCapture cap;
    cv::dnn::Net net;
    std::vector<std::string> classNames;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    
    void loadModel();
    QImage cvMatToQImage(const cv::Mat &inMat);
    
    // 添加统一的检测处理函数
    void processDetection(const cv::Mat& input_frame, cv::Mat& output_frame);
    
    // 添加检测相关的常量
    const float CONFIDENCE_THRESHOLD = 0.15f;  // 置信度阈值
    const float NMS_THRESHOLD = 0.4f;         // NMS阈值
    const int MODEL_WIDTH = 640;              // 模型输入宽度
    const int MODEL_HEIGHT = 640;             // 模型输入高度

    // 生产者消费者相关函数
    void producerThread();
    void consumerThread();
    void startThreads();
    void stopThreads();
    
    // 线程控制
    std::atomic<bool> running{false};
    std::thread producer;
    std::thread consumer;
    
    // 预处理后的图像队列
    struct ProcessedFrame {
        cv::Mat original;
        cv::Mat blob;
        std::chrono::system_clock::time_point timestamp;
    };
    SafeQueue<ProcessedFrame> frameQueue;
    
    // 处理后的结果队列
    struct DetectionResult {
        cv::Mat frame;
        std::chrono::system_clock::time_point timestamp;
    };
    SafeQueue<DetectionResult> resultQueue;
};

#endif // MAINWINDOW_H
