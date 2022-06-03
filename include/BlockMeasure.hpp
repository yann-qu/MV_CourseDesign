#pragma once

#include "opencv2/opencv.hpp"
#include <memory>

class BlockMeasure {
 public:
  BlockMeasure() = default;
  BlockMeasure(cv::Mat& img_) : pimg(std::make_shared<cv::Mat>(img_)) {}
  void init(cv::Mat& img_);

  ~BlockMeasure() = default;

 private:
  std::shared_ptr<cv::Mat> pimg;
  
};
