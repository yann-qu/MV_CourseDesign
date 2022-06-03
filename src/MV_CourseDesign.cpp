#include "MV_CourseDesign.hpp"

int main(const int argc, const char **argv) {
  cv::Mat src;
  BlockMeasure bm;
  src = cv::imread("E:/Code/C_Cpp_practices/MV_CourseDesign/resource/xx-grating_002_0.bmp");
  bm.init(src);
#ifdef DEBUG
  cv::imshow("src", src);
  cv::waitKey(0);
#endif
  return 0;
}
