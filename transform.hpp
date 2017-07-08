#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

class Transform {
public:
    static cv::Mat four_point_transform(cv::Mat const &image, std::vector<cv::Point> const &points);
};
