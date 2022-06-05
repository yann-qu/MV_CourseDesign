#include "BlockMeasure.hpp"

// 定义ROI 对应八个条纹
std::vector<std::vector<cv::Point>> ROIdef = {
  // 1
  {cv::Point(Stripe1LUX, YStart),
    cv::Point(Stripe1LUX + ROIHORWidth, YStart + ROIHORHeight1),
    cv::Point(Stripe1LUX, YStart + Stripe1L),
    cv::Point(Stripe1LUX + ROIHORWidth, YStart + Stripe1L + ROIHORHeight2),

    cv::Point(Stripe1LUX - GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe1LUX - GETStripeVER1 + ROIVERWidth, YStart + ROIVERHeight),
    cv::Point(Stripe1LUX + GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe1LUX + GETStripeVER1 + ROIVERWidth,
              YStart + ROIVERHeight)},
  // 2
  {cv::Point(Stripe2LUX, YStart),
    cv::Point(Stripe2LUX + ROIHORWidth, YStart + ROIHORHeight1),
    cv::Point(Stripe2LUX, YStart + Stripe2L),
    cv::Point(Stripe2LUX + ROIHORWidth, YStart + Stripe2L + ROIHORHeight2),

    cv::Point(Stripe2LUX - GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe2LUX - GETStripeVER1 + ROIVERWidth, YStart + ROIVERHeight),
    cv::Point(Stripe2LUX + GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe2LUX + GETStripeVER1 + ROIVERWidth,
              YStart + ROIVERHeight)},
  // 3
  {cv::Point(Stripe3LUX, YStart),
    cv::Point(Stripe3LUX + ROIHORWidth, YStart + ROIHORHeight1),
    cv::Point(Stripe3LUX, YStart + Stripe3L),
    cv::Point(Stripe3LUX + ROIHORWidth, YStart + Stripe3L + ROIHORHeight2),

    cv::Point(Stripe3LUX - GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe3LUX - GETStripeVER1 + ROIVERWidth, YStart + ROIVERHeight),
    cv::Point(Stripe3LUX + GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe3LUX + GETStripeVER1 + ROIVERWidth,
              YStart + ROIVERHeight)},
  // 4
  {cv::Point(Stripe4LUX, YStart),
    cv::Point(Stripe4LUX + ROIHORWidth, YStart + ROIHORHeight1),
    cv::Point(Stripe4LUX, YStart + Stripe4L),
    cv::Point(Stripe4LUX + ROIHORWidth, YStart + Stripe4L + ROIHORHeight2),

    cv::Point(Stripe4LUX - GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe4LUX - GETStripeVER1 + ROIVERWidth, YStart + ROIVERHeight),
    cv::Point(Stripe4LUX + GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe4LUX + GETStripeVER1 + ROIVERWidth,
              YStart + ROIVERHeight)},
  // 5
  {cv::Point(Stripe5LUX, YStart),
    cv::Point(Stripe5LUX + ROIHORWidth, YStart + ROIHORHeight1),
    cv::Point(Stripe5LUX, YStart + Stripe5L),
    cv::Point(Stripe5LUX + ROIHORWidth, YStart + Stripe5L + ROIHORHeight2),

    cv::Point(Stripe5LUX - GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe5LUX - GETStripeVER1 + ROIVERWidth, YStart + ROIVERHeight),
    cv::Point(Stripe5LUX + GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe5LUX + GETStripeVER1 + ROIVERWidth,
              YStart + ROIVERHeight)},
  // 6
  {cv::Point(Stripe6LUX, YStart),
    cv::Point(Stripe6LUX + ROIHORWidth, YStart + ROIHORHeight1),
    cv::Point(Stripe6LUX, YStart + Stripe6L),
    cv::Point(Stripe6LUX + ROIHORWidth, YStart + Stripe6L + ROIHORHeight2),

    cv::Point(Stripe6LUX - GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe6LUX - GETStripeVER1 + ROIVERWidth, YStart + ROIVERHeight),
    cv::Point(Stripe6LUX + GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe6LUX + GETStripeVER1 + ROIVERWidth,
              YStart + ROIVERHeight)},
  // 7
  {cv::Point(Stripe7LUX, YStart),
    cv::Point(Stripe7LUX + ROIHORWidth, YStart + ROIHORHeight1),
    cv::Point(Stripe7LUX, YStart + Stripe7L),
    cv::Point(Stripe7LUX + ROIHORWidth, YStart + Stripe7L + ROIHORHeight2),

    cv::Point(Stripe7LUX - GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe7LUX - GETStripeVER1 + ROIVERWidth, YStart + ROIVERHeight),
    cv::Point(Stripe7LUX + GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe7LUX + GETStripeVER1 + ROIVERWidth,
              YStart + ROIVERHeight)},
  // 8
  {cv::Point(Stripe8LUX, YStart),
    cv::Point(Stripe8LUX + ROIHORWidth, YStart + ROIHORHeight1),
    cv::Point(Stripe8LUX, YStart + Stripe8L),
    cv::Point(Stripe8LUX + ROIHORWidth, YStart + Stripe8L + ROIHORHeight2),

    cv::Point(Stripe8LUX - GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe8LUX - GETStripeVER1 + ROIVERWidth, YStart + ROIVERHeight),
    cv::Point(Stripe8LUX + GETStripeVER1, YStart + GETStripeVER2),
    cv::Point(Stripe8LUX + GETStripeVER1 + ROIVERWidth,
              YStart + ROIVERHeight)},
};

/**
 * @brief 初始化一个BlockMeasure对象
 * @param img_
 */
void BlockMeasure::Init(cv::Mat &img_) {
  this->img_origin = img_;
  this->img = img_.clone();
}

/**
 * @brief 对图片进行旋转矫正
 */
void BlockMeasure::Rotate() {
  cv::Point2d Origin(this->img_origin.cols / 2.0, this->img_origin.rows / 2.0);
  MyDFT(this->img_origin, this->img_DFT);
  double angelR = MyRotateAngle(this->img_DFT);
  cv::Mat RotateMat = cv::getRotationMatrix2D(Origin, angelR - 90, 1.0);
  cv::warpAffine(this->img, this->img, RotateMat, this->img.size(),
                 cv::INTER_LINEAR | cv::WARP_FILL_OUTLIERS,
                 cv::BORDER_REPLICATE, cv::Scalar(255, 255, 255));
}

/**
 * @brief 对图片进行滤波处理，降低图片噪声
 */
void BlockMeasure::Filter() {
  cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

  // 高斯滤波去除高斯噪声
  // sigmaX和sigmaY为3时方差最小
  cv::GaussianBlur(this->img, this->img, cv::Size(5, 5), 3, 3);

  cv::morphologyEx(this->img, this->img, cv::MORPH_OPEN, element);

  for (int i = 0; i < this->img.rows; i++) {
    for (int j = 0; j < this->img.cols; j++) {
      if (this->img.at<uchar>(i, j) >= 130) {
        this->img.at<uchar>(i, j) = 255;
      } else if (this->img.at<uchar>(i, j) < 100 && this->img.at<uchar>(i, j) >= 50) {
        this->img.at<uchar>(i, j) = this->img.at<uchar>(i, j) - 50;
      }
      if (this->img.at<uchar>(i, j) < 50 && this->img.at<uchar>(i, j) >= 10) {
        this->img.at<uchar>(i, j) = 10;
      }
    }
  }
  // TODO 对比二值化和多阈值处理
  // cv::threshold(*(this->pimg), *(this->pimg), 0, 255, cv::THRESH_OTSU);
}


/**
 * @brief 对图片进行亚像素精度的1D测量
 */
void BlockMeasure::Measure() {
  int StripeIndex = 0;  // Stripe index, from 0 to 7. There are 8 in total.
  // subPixel是一个存储了4个vector的数组，每个vector分别存储了上下左右四个边的ROI中边缘的坐标
  // PresubPixel记录上个条纹的右侧边缘， 用于计算条纹间隔。
  std::vector<cv::Point2d> subPixel[4], PresubPixel;
  cv::Mat ROI;  // region of interest
  std::vector<double> stripes_width, gaps_width;
  for (; StripeIndex < 8; StripeIndex++) {
    for (int k = 0; k < 4; k++) {
      subPixel[k].clear();
      // 把ROI截取出来
      selectROI(this->img, ROI, ROIdef[StripeIndex][2 * k], ROIdef[StripeIndex][2 * k + 1]);
      // 在ROI中进行亚像素级边缘提取
      extractSubPixel(ROI, subPixel[k], 30, k < 2, !(k & 1), SampleAmount, kSize);
      // 将边缘相对于ROI的坐标变换到相对于整幅图的坐标。
      for (int cnt = 0; cnt < subPixel[k].size(); cnt++) {
        subPixel[k].at(cnt).x += ROIdef[StripeIndex][2 * k].x;
        subPixel[k].at(cnt).y += ROIdef[StripeIndex][2 * k].y;
      }
    }
    double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    for (int i = 0; i < SampleAmount; i++) {
      sum1 += (subPixel[1][i].y - subPixel[0][i].y);
      sum2 += (subPixel[3][i].x - subPixel[2][i].x);
      if (!PresubPixel.empty()) sum3 += (subPixel[2][i].x - PresubPixel[i].x);
    }

    this->stripe_length.emplace_back(sum1 / SampleAmount);
    stripes_width.emplace_back(sum2 / SampleAmount);

    if (StripeIndex > 0)
      gaps_width.emplace_back(sum3 / SampleAmount);
    PresubPixel.clear();
    for (int j = 0; j < SampleAmount; j++)
      PresubPixel.emplace_back(subPixel[3][j]);
  }

  double sum2 = 0.0, sum3 = 0.0;
  for (int i = 0; i < 8; i++) {
    sum2 += stripes_width[i];
    if (i < 7) sum3 += gaps_width[i];
  }
  this->stripe_width = sum2 / 8;
  this->gap_width = sum3 / 7;
}

/**
 * @brief 打印结果，显示图片
 */
void BlockMeasure::Display() {
  char buffer[64];
  memset(buffer, 0, sizeof(buffer));
  sprintf(buffer, "stripe width = %.2f, gap width = %.2f, time usage = %.2lfms", this->stripe_width, this->gap_width,
          this->time);
  std::cout << buffer << std::endl;
  cv::putText(this->img_origin, buffer,
              cv::Point(this->img_origin.cols / 20, this->img_origin.rows / 15 + 40),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(72, 73, 220), 2,
              cv::LINE_AA);


  memset(buffer, 0, sizeof(buffer));
  sprintf(buffer, "stripe length = %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f", this->stripe_length[0],
          this->stripe_length[1], this->stripe_length[2], this->stripe_length[3], this->stripe_length[4],
          this->stripe_length[5], this->stripe_length[6], this->stripe_length[7]);
  std::cout << buffer << std::endl;
  cv::putText(this->img_origin, buffer,
              cv::Point(this->img_origin.cols / 20, this->img_origin.rows / 15 + 80),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(72, 73, 220), 2,
              cv::LINE_AA);

  cv::imshow("img", this->img_origin);
  cv::waitKey(0);
}

void BlockMeasure::Process(cv::Mat &img_) {
  this->Init(img_);
  int64_t t1 = cv::getTickCount();
  this->Rotate();
  this->Filter();
  this->Measure();
  int64_t t2 = cv::getTickCount();
  this->time = 1.0 * (t2 - t1) * 1000 / cv::getTickFrequency();
}


/**
 * @brief 对图片进行离散傅里叶变换
 * @param src
 * @param dst
 */
void MyDFT(cv::InputArray &src, cv::OutputArray &dst) {
  cv::Size dftSize;
  // calculate the size of DFT transform
  dftSize.width = cv::getOptimalDFTSize(src.cols());
  dftSize.height = cv::getOptimalDFTSize(src.rows());

  cv::Mat temp;
  copyMakeBorder(src, temp, 0, dftSize.height - src.rows(), 0,
                 dftSize.width - src.cols(), cv::BORDER_CONSTANT,
                 cv::Scalar::all(0));
  cv::Mat planes[] = {cv::Mat_<float>(temp),
                      cv::Mat::zeros(temp.size(), CV_32F)};
  cv::Mat ComplexI;
  merge(planes, 2, ComplexI);  // Add to the expanded another plane with zeros
  cv::dft(ComplexI, ComplexI);
  split(ComplexI, planes);  // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
  magnitude(planes[0], planes[1], planes[0]);  // planes[0] = magnitude
  cv::Mat FourierFrame = planes[0];

  FourierFrame += cv::Scalar::all(1);  // switch to logarithmic scale
  //计算幅值，转换到对数尺度(logarithmic scale)
  cv::log(FourierFrame, FourierFrame);  //转换到对数尺度(logarithmic scale)

  //如果有奇数行或列，则对频谱进行裁剪
  FourierFrame = FourierFrame(
    cv::Rect(0, 0, FourierFrame.cols & -2, FourierFrame.rows & -2));

  //重新排列傅里叶图像中的象限，使得原点位于图像中心
  int cx = FourierFrame.cols / 2;
  int cy = FourierFrame.rows / 2;

  cv::Mat q0(FourierFrame, cv::Rect(0, 0, cx, cy));  //左上角图像划定ROI区域
  cv::Mat q1(FourierFrame, cv::Rect(cx, 0, cx, cy));   //右上角图像
  cv::Mat q2(FourierFrame, cv::Rect(0, cy, cx, cy));   //左下角图像
  cv::Mat q3(FourierFrame, cv::Rect(cx, cy, cx, cy));  //右下角图像

  //变换左上角和右下角象限
  cv::Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  //变换右上角和左下角象限
  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  //归一化处理，用0-1之间的浮点数将矩阵变换为可视的图像格式
  normalize(FourierFrame, FourierFrame, 0, 255, cv::NORM_MINMAX);
  FourierFrame.copyTo(dst);

}

/**
 * @brief 从频域图像中提取偏离角度s
 * @param src
 * @return
 */
double MyRotateAngle(cv::InputArray src) {
  cv::Mat frameOrigin, m_frameOrigin, frameGrey, frameThreshold, tmp;
  std::vector<double> lineAngle;
  double Theta = 0, Rho, Angel;

  src.copyTo(frameOrigin);
  frameOrigin.convertTo(frameGrey, CV_8UC1);

  cv::threshold(frameGrey, frameThreshold, 145, 255, cv::THRESH_BINARY);

  std::vector<cv::Vec2f> lines;
  cv::HoughLines(frameThreshold, lines, 1, CV_PI / 180, 20, 10, 10, 1.4,
                 CV_PI / 2);  // runs the actual detection
  // todo 后面可以换成HoughLineP， 据说速度更快

  Rho = lines[0][0], Theta = lines[0][1];

  cv::Point pt1, pt2;
  double a = cos(Theta), b = sin(Theta);
  double x0 = 1.0 * frameOrigin.cols / 2, y0 = 1.0 * frameOrigin.rows / 2;
  pt1.x = cvRound(x0 + 1000 * (-b));
  pt1.y = cvRound(y0 + 1000 * (a));
  pt2.x = cvRound(x0 - 1000 * (-b));
  pt2.y = cvRound(y0 - 1000 * (a));

  if (Theta != CV_PI / 2) {
    double angelT = (float) (frameGrey.rows * tan(Theta) / frameGrey.cols);
    Theta = (float) atan(angelT);
  }
  Angel = Theta * 180 / (float) CV_PI;

  return Angel;
}

/**
 * @brief 函数根据给出的两个角点（左上和右下）， 在输入图像响应坐标位置截取矩形区域
 * @param src
 * @param dst
 * @param leftUpper
 * @param rightLower
 */
void selectROI(cv::Mat &src, cv::Mat &dst, cv::Point &leftUpper, cv::Point &rightLower) {
  cv::Rect roi = cv::Rect(leftUpper, rightLower);
  dst = src(roi).clone();
}

bool cmp(cv::Point2d a, cv::Point2d b) { return a.y > b.y; }

bool rcmp(cv::Point2d a, cv::Point2d b) { return a.y < b.y; }

void extractSubPixel(cv::Mat &src, std::vector<cv::Point2d> &subPixel, int threshold, bool IsHorizontal, bool Ascending,
                     int SampleAmount, int kSize) {
  int h = src.rows, w = src.cols;  // Src is ROI passed in.
  int len = IsHorizontal ? w : h;
  int range = IsHorizontal ? h : w;
  int sampleGap = len / SampleAmount;
  int border = (kSize - 1) / 2;

  std::vector<double> test;
  cv::Mat srcGrey;
  cv::Mat temp = src.clone();
  if (src.channels() == 3) cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
  double a, b, c;
  cv::Mat GaussianKernel;

  std::vector<cv::Point2d> firstDerivative;

  Gen_GaussianKernel(GaussianKernel, kSize, 5);
  for (int delta = sampleGap; delta < len; delta += sampleGap) {
      firstDerivative.clear();
      for (int i = border; i < -border + range; i++) {
      double sum = 0.0;
      for (int j = -border; j <= border; j++) {
        if (IsHorizontal) {
          sum += (GaussianKernel.at<double>(0, border + j) *
                  src.at<uchar>(i + j, delta));
        } else {
          sum += (GaussianKernel.at<double>(0, border + j) *
                  src.at<uchar>(delta, i + j));
        }
      }
      if ((Ascending && sum - threshold > EPS) || !Ascending && -sum - threshold > EPS)
        firstDerivative.emplace_back(cv::Point2d(i, sum));  //结果大于阈值才压入数组
    }
    if (Ascending)
      std::sort(firstDerivative.begin(), firstDerivative.end(),
                cmp);  //对结果进行降序排序， 以便获得三个最大的点。
    else
      std::sort(firstDerivative.begin(), firstDerivative.end(),
                rcmp);  //对结果进行升序排序， 以便获得三个最小的点。
    if (fitParabola(firstDerivative, a, b, c)) {
      if (IsHorizontal)
        subPixel.emplace_back(
          cv::Point2d(delta, -b / (2 * a)));  // 将亚像素点压入数组
      else
        subPixel.emplace_back(cv::Point2d(-b / (2 * a), delta));
    }
  }
}

/**
 * @brief 使用高斯滤波器的一阶导数产生最优边缘滤波器。
 * @param OutputArray
 * @param kSize
 * @param sigma
 */
void Gen_GaussianKernel(cv::Mat &OutputArray, int kSize, int sigma) {
  OutputArray = cv::Mat::zeros(1, kSize, CV_64F);
  std::vector<double> DataInsight;

  int center = (kSize - 1) / 2;
  double x, z;

  for (int j = 0; j < kSize; j++) {
    x = pow(j - center, 2);
    z = -(j - center) * exp(-(x) / (2 * sigma * sigma));
    OutputArray.at<double>(0, j) = z;
  }
}

/**
 * @brief 拟合抛物线方程，ax^2 + bx + c = y，计算a b c的值
 * @param vecPoints
 * @param a
 * @param b
 * @param c
 * @return
 */
bool fitParabola(const std::vector<cv::Point2d> &vecPoints, double &a, double &b, double &c) {
  if (vecPoints.size() < 3) return false;
  // 初始化 Mat
  cv::Mat matA(3, 3, CV_64F);
  cv::Mat matB(3, 1, CV_64F);
  cv::Mat matC(3, 1, CV_64F);

  // 构造元素
  for (int i = 0; i < 3; ++i) {
    matA.at<double>(i, 0) = vecPoints[i].x * vecPoints[i].x;
    matA.at<double>(i, 1) = vecPoints[i].x;
    matA.at<double>(i, 2) = 1;
  }
  for (int i = 0; i < 3; ++i) {
    matB.at<double>(i, 0) = vecPoints[i].y;
  }
  // opencv最小二乘法拟合
  cv::solve(matA, matB, matC, cv::DECOMP_LU);
  // 返回值
  a = matC.at<double>(0, 0);
  b = matC.at<double>(0, 1);
  c = matC.at<double>(0, 2);
  return true;
}




