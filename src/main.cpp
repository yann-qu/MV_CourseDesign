#include "MV_CourseDesign.hpp"

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
