#include "MV_CourseDesign.hpp"

std::pair<double, double> cal_mean_var(std::vector<double> &vec) {
  double sum = std::accumulate(std::begin(vec), std::end(vec), 0.0);
  double mean = sum / vec.size(); // 计算均值
  double var = 0.0;
  std::for_each(std::begin(vec), std::end(vec), [&](const double d) { var += (d - mean) * (d - mean); });  // 计算方差
  return {mean, var / vec.size()};
}

void evaluate(std::vector<double> &stripes_width, std::vector<double> &gaps_width) {
  std::pair<double, double> p_evaluate;
  p_evaluate = cal_mean_var(stripes_width);
  std::cout << "mean of stripes width = " << p_evaluate.first << " variance of stripes width = " << p_evaluate.second << std::endl;
  p_evaluate = cal_mean_var(gaps_width);
  std::cout << "mean of gap width = " << p_evaluate.first << " variance of gap width = " << p_evaluate.second << std::endl;
  auto axes = CvPlot::makePlotAxes();
  axes.create<CvPlot::Series>(stripes_width, "-go");
  axes.create<CvPlot::Series>(gaps_width, "-bo");
  CvPlot::show("img", axes);
  cv::waitKey(0);
  cv::destroyAllWindows();
}

