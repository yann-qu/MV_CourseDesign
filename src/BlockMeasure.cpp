#include "BlockMeasure.hpp"

void BlockMeasure::init(cv::Mat& img_) {
  this->pimg = std::make_shared<cv::Mat>(img_);
}


