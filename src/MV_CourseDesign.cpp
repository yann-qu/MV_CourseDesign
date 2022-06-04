#include "MV_CourseDesign.hpp"

std::string res_path = "../resource/";
std::vector<std::string> res_name = {
  "xx-grating_002_0.bmp",
  "noised_0.01.bmp",
  "noised_0.03.bmp",
  "noised_0.05.bmp",
  "noised_0.07.bmp",
  "noised_0.09.bmp",
  "noised_0.1.bmp",
  "noised_0.3.bmp",
  "noised_0.5.bmp"
};

int main(const int argc, const char **argv) {
  BlockMeasure bm;
  for (auto&& i : res_name) {
    cv::Mat src = cv::imread(res_path + i, cv::IMREAD_GRAYSCALE);
    bm.Process(src);
    bm.Display();

  }
  return 0;
}
