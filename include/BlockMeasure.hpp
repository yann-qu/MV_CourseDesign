#pragma once

#include "opencv2/opencv.hpp"
#include <algorithm>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdio>

#define YStart 273
#define ROIHORWidth 20
#define ROIHORHeight1 15
#define ROIHORHeight2 230
#define ROIVERWidth 24
#define ROIVERHeight 150

#define GETStripeVER1 15
#define GETStripeVER2 25

// #define Stripe1LUX 275
#define Stripe1LUX 280
#define Stripe1L 185
#define Stripe2LUX 335
#define Stripe2L 185
#define Stripe3LUX 395
#define Stripe3L 185
#define Stripe4LUX 455
#define Stripe4L 185
#define Stripe5LUX 515
#define Stripe5L 185
#define Stripe6LUX 575
#define Stripe6L 185
#define Stripe7LUX 635
#define Stripe7L 185
#define Stripe8LUX 695
#define Stripe8L 185

#define EPS 1e-4

extern std::vector<std::vector<cv::Point>> ROIdef;
const int SampleAmount = 20;
const int kSize = 5;

class BlockMeasure {
public:
  BlockMeasure() = default;
  BlockMeasure(cv::Mat &img_) : img_origin(img_), img(img_.clone()) {}
  void Process(cv::Mat &img_);
  void Display();
  ~BlockMeasure() = default;

private:
  void Init(cv::Mat &img_);
  void Filter();
  void Rotate();
  void Measure();

  // std::shared_ptr<cv::Mat> pimg;
  // std::shared_ptr<cv::Mat> pimg_origin;
  double time;
  cv::Mat img;         // 存储处理的图像
  cv::Mat img_origin;  // 存储原图像
  cv::Mat img_DFT;     // 存储DFT结果
  double stripe_width;   // 存储物体宽度
  double gap_width;      // 存储间隔宽度
  std::vector<double> stripe_length;  // 存储物体底端到顶端的长度
};


void MyDFT(cv::InputArray &src, cv::OutputArray &dst);
double MyRotateAngle(cv::InputArray src);
void selectROI(cv::Mat &src, cv::Mat &dst, cv::Point &leftUpper, cv::Point &rightLower);
void extractSubPixel(cv::Mat &src, std::vector<cv::Point2d> &subPixel, int threshold, bool IsHorizontal = true, bool Ascending = true, int SampleAmount = 20, int kSize = 5);
void Gen_GaussianKernel(cv::Mat &OutputArray, int kSize, int sigma);
bool fitParabola(const std::vector<cv::Point2d> &vecPoints, double &a, double &b, double &c);
