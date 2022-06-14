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
  std::vector<double> stripes_width, gaps_width;
  for (auto &&i: res_name) {
    StripeMeasure sm;
    cv::Mat src = cv::imread(res_path + i, cv::IMREAD_GRAYSCALE);
    sm.Process(src);
    sm.Display();
    stripes_width.emplace_back(sm.get_stripe_width());
    gaps_width.emplace_back(sm.get_gap_width());
  }
  evaluate(stripes_width, gaps_width);
  return 0;
}
