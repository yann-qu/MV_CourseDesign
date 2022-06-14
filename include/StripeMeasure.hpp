#pragma once

#include "opencv2/opencv.hpp"
#include <algorithm>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdio>

#define YStart 273
#define ROIHorWidth 20
#define ROIHorHeight1 15
#define ROIHorHeight2 230
#define ROIVerWidth 24
#define ROIVerHeight 150

#define GETStripeVER1 15
#define GETStripeVER2 25

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

extern std::vector<std::vector<cv::Point>> ROIDiagPoints;
const int SampleAmount = 20;
const int kSize = 5;

class StripeMeasure {
public:
  StripeMeasure() = default;
  StripeMeasure(cv::Mat &img_) : img_origin(img_), img(img_.clone()) {}
  void Process(cv::Mat &img_);
  void Display();
  double get_stripe_width() const { return this->stripe_width; }
  double get_gap_width() const { return this->gap_width; }
  ~StripeMeasure() = default;

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


void DFT(cv::InputArray &src, cv::OutputArray &dst);
double RotateAngle(cv::InputArray src);
void SelectROI(cv::Mat &src, cv::Mat &dst, cv::Point &leftUpper, cv::Point &rightLower);
void ExtractSubPixel(cv::Mat &ROI, std::vector<cv::Point2d> &subPixel, int threshold, bool isHorizontal = true,
                     bool ascending = true, int sampleAmount = 20, int kSize = 5);
void GenGaussianKernel(cv::Mat &outputArray, int kSize, int sigma);
bool FitPara(const std::vector<cv::Point2d> &vecPoints, double &a, double &b, double &c);
