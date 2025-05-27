#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QPainter>
#include <QDebug>
#include <cmath>
#include <fstream>
#include <thread>
#include <chrono>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::updateFrame);

    qDebug() << "正在初始化...";

    // 检查模型文件是否存在
    QFileInfo modelFile("E:/qt/yolo11n.onnx");
    if (!modelFile.exists()) {
        QMessageBox::critical(this, "错误", "找不到模型文件：" + modelFile.absoluteFilePath());
        return;
    }
    qDebug() << "模型文件存在：" << modelFile.absoluteFilePath();

    // 检查类别文件是否存在
    QFileInfo namesFile("E:/qt/coco.names");
    if (!namesFile.exists()) {
        QMessageBox::critical(this, "错误", "找不到类别文件：" + namesFile.absoluteFilePath());
        return;
    }
    qDebug() << "类别文件存在：" << namesFile.absoluteFilePath();

    loadModel();
}

MainWindow::~MainWindow()
{
    if(cap.isOpened()) {
        cap.release();
    }
    delete ui;
}

void MainWindow::loadModel()
{
    try {
        qDebug() << "开始加载模型...";

        // 尝试加载模型文件
        try {
            net = cv::dnn::readNet("E:/qt/yolo11n.onnx");
            qDebug() << "模型文件加载完成";
        }
        catch (const cv::Exception& e) {
            QMessageBox::critical(this, "错误", "模型加载失败(OpenCV异常)：" + QString::fromStdString(e.msg));
            qDebug() << "模型加载失败：" << QString::fromStdString(e.msg);
            return;
        }

        if (net.empty()) {
            QMessageBox::critical(this, "错误", "模型加载失败：模型为空");
            qDebug() << "模型加载失败：模型为空";
            return;
        }

        // 设置计算后端和目标设备
        try {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            qDebug() << "模型后端和目标设备设置完成";
        }
        catch (const cv::Exception& e) {
            QMessageBox::critical(this, "错误", "模型配置失败：" + QString::fromStdString(e.msg));
            qDebug() << "模型配置失败：" << QString::fromStdString(e.msg);
            return;
        }

        // 读取类别名称
        try {
            std::ifstream ifs("E:/qt/coco.names");
            if (!ifs.is_open()) {
                QMessageBox::critical(this, "错误", "无法打开类别文件：E:/qt/coco.names");
                qDebug() << "无法打开类别文件";
                return;
            }

            classNames.clear();
            std::string line;
            while (std::getline(ifs, line)) {
                if (!line.empty()) {
                    classNames.push_back(line);
                }
            }
            qDebug() << "加载了" << classNames.size() << "个类别";

            if (classNames.empty()) {
                QMessageBox::warning(this, "警告", "类别列表为空，请检查coco.names文件内容");
                qDebug() << "类别列表为空";
            }
        }
        catch (const std::exception& e) {
            QMessageBox::critical(this, "错误", "读取类别文件时发生异常：" + QString::fromStdString(e.what()));
            qDebug() << "读取类别文件异常：" << QString::fromStdString(e.what());
            return;
        }

        qDebug() << "模型初始化完成";
    }
    catch (const std::exception& e) {
        QMessageBox::critical(this, "错误", "发生异常：" + QString::fromStdString(e.what()));
        qDebug() << "发生异常：" << QString::fromStdString(e.what());
    }
}

void MainWindow::producerThread() {
    while (running) {
        cv::Mat frame;
        if (!cap.read(frame)) {
            qDebug() << "无法读取视频帧";
            running = false;
            break;
        }

        // 预处理
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1./255., cv::Size(MODEL_WIDTH, MODEL_HEIGHT),
                              cv::Scalar(0,0,0), true, false);

        // 创建处理后的帧对象
        ProcessedFrame processedFrame{
            frame,
            blob,
            std::chrono::system_clock::now()
        };

        // 将预处理后的帧放入队列
        if (!frameQueue.try_push(std::move(processedFrame))) {
            qDebug() << "帧队列已满，丢弃旧帧";
            ProcessedFrame processedFrame_;
            frameQueue.try_pop(processedFrame_);
            frameQueue.try_push(std::move(processedFrame));
        }
    }
}

void MainWindow::consumerThread() {
    while (running) {
        ProcessedFrame processedFrame;
        if (!frameQueue.try_pop(processedFrame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        cv::Mat result;
        processDetection(processedFrame.original, result);

        // 将处理后的结果放入结果队列
        DetectionResult detection_result{
            result,
            processedFrame.timestamp
        };

        if (!resultQueue.try_push(std::move(detection_result))) {
            qDebug() << "结果队列已满，丢弃旧结果";
            DetectionResult detection_result_;
            resultQueue.try_pop(detection_result_);
            resultQueue.try_push(std::move(detection_result));
        }
    }
}

void MainWindow::startThreads() {
    running = true;
    producer = std::thread(&MainWindow::producerThread, this);
    consumer = std::thread(&MainWindow::consumerThread, this);
}

void MainWindow::stopThreads() {
    running = false;
    if (producer.joinable()) {
        producer.join();
    }
    if (consumer.joinable()) {
        consumer.join();
    }
    frameQueue.clear();
    resultQueue.clear();
}

void MainWindow::updateFrame() {
    DetectionResult result;
    if (resultQueue.try_pop(result)) {
        // 计算延迟
        auto now = std::chrono::system_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - result.timestamp).count();
        qDebug() << "处理延迟:" << latency << "ms";

        // 转换为QImage并显示
        QImage image = cvMatToQImage(result.frame);
        if (!image.isNull()) {
            ui->label->setPixmap(QPixmap::fromImage(image).scaled(
                ui->label->size(), Qt::KeepAspectRatio));
        }
    }
}


QImage MainWindow::cvMatToQImage(const cv::Mat &inMat)
{
    switch (inMat.type()) {
        case CV_8UC4: {
            QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_ARGB32);
            return image.copy();
        }
        case CV_8UC3: {
            QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_RGB888);
            return image.rgbSwapped();
        }
        case CV_8UC1: {
            QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_Grayscale8);
            return image.copy();
        }
        default:
            return QImage();
    }
}

void MainWindow::on_startButton_clicked()
{
    if (!cap.isOpened()) {
        cap.open(0);
        if (!cap.isOpened()) {
            QMessageBox::warning(this, "错误", "无法打开摄像头");
            return;
        }
    }
    startThreads();
    timer->start(30); // 30ms 刷新率
}

void MainWindow::on_stopButton_clicked()
{
    timer->stop();
    stopThreads();
    if (cap.isOpened()) {
        cap.release();
    }
    ui->label->clear();
}

void MainWindow::processDetection(const cv::Mat& input_frame, cv::Mat& output_frame) {
    try {
        // 预处理
        cv::Mat blob;
        cv::dnn::blobFromImage(input_frame, blob, 1./255., cv::Size(MODEL_WIDTH, MODEL_HEIGHT), cv::Scalar(0,0,0), true, false);

        // 设置网络输入
        net.setInput(blob);

        // 推理
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        if (outputs.empty() || outputs[0].empty()) {
            qDebug() << "模型输出为空!";
            output_frame = input_frame;
            return;
        }

        // 处理YOLOv8输出
        cv::Mat output = outputs[0];
        cv::Mat transposed = output.reshape(1, output.size[1]); // 84 x 8400
        transposed = transposed.t();  // 8400 x 84

        // 清空之前的检测结果
        boxes.clear();
        confidences.clear();
        classIds.clear();

        // 遍历每个检测框
        for (int i = 0; i < transposed.rows; ++i) {
            float* row = (float*)transposed.ptr(i);

            // 获取类别得分
            cv::Mat scores(1, classNames.size(), CV_32F, row + 4);
            cv::Point class_id;
            double confidence;
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id);

            // 使用类别得分作为置信度
            if (confidence > CONFIDENCE_THRESHOLD) {
                // 获取边界框坐标
                float x = row[0];
                float y = row[1];
                float w = row[2];
                float h = row[3];

                int left = static_cast<int>((x - w/2));
                int top = static_cast<int>((y - h/2));
                int width = static_cast<int>(w);
                int height = static_cast<int>(h);

                // // 确保坐标和尺寸在有效范围内
                left = std::max(0, std::min(input_frame.cols - 1, left));
                top = std::max(0, std::min(input_frame.rows - 1, top))-80;
                width = std::max(1, std::min(input_frame.cols - left, width));
                height = std::max(1, std::min(input_frame.rows - top, height));

                // 保存检测结果
                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(static_cast<float>(confidence));
                classIds.push_back(class_id.x);
            }
        }

        // 执行NMS
        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

            // 绘制检测结果
            output_frame = input_frame.clone();
            for (size_t i = 0; i < indices.size(); ++i) {
                int idx = indices[i];
                const cv::Rect& box = boxes[idx];
                int classId = classIds[idx];
                float conf = confidences[idx];

                // 绘制边界框
                cv::rectangle(output_frame, box, cv::Scalar(0, 255, 0), 2);

                // 准备标签文本
                std::string label = classNames[classId] + ": " +
                                  std::to_string(static_cast<int>(conf * 100)) + "%";

                // 绘制标签背景
                int baseline = 0;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                   0.5, 1, &baseline);
                cv::rectangle(output_frame,
                            cv::Point(box.x, box.y - labelSize.height - baseline - 5),
                            cv::Point(box.x + labelSize.width, box.y),
                            cv::Scalar(0, 255, 0), cv::FILLED);

                // 绘制标签文本
                cv::putText(output_frame, label,
                           cv::Point(box.x, box.y - baseline - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        } else {
            output_frame = input_frame.clone();
        }
    }
    catch (const cv::Exception& e) {
        qDebug() << "OpenCV异常：" << QString::fromStdString(e.msg);
        output_frame = input_frame;
    }
    catch (const std::exception& e) {
        qDebug() << "标准异常：" << QString::fromStdString(e.what());
        output_frame = input_frame;
    }
    catch (...) {
        qDebug() << "未知异常！";
        output_frame = input_frame;
    }
}
