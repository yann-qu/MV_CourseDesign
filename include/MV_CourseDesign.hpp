#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include "opencv2/opencv.hpp"
#include "StripeMeasure.hpp"

#define CVPLOT_HEADER_ONLY
#include "CvPlot/cvplot.h"

extern std::string res_path;
extern std::vector<std::string> res_name;

std::pair<double, double> cal_mean_var(std::vector<double> &vec);
void evaluate(std::vector<double> &stripes_width, std::vector<double> &gaps_width);

