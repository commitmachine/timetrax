#include "transform.hpp"
#include <math.h>
#include <algorithm>
using namespace std;

cv::Mat Transform::four_point_transform(cv::Mat const &image, std::vector<cv::Point> const &points) {
    // TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT
    cv::Point tl = points[0];
    cv::Point tr = points[1];
    cv::Point br = points[2];
    cv::Point bl = points[3];
    double widthA = sqrt( pow((br.x - bl.x), 2) + pow((br.y - bl.y), 2) );
    double widthB = sqrt( pow((tr.x - tl.x), 2) + pow((tr.y - tl.y), 2) );
    double heightA = sqrt( pow((tr.x - br.x), 2) + pow((tr.y - br.y), 2) );
    double heightB = sqrt( pow((tl.x - bl.x), 2) + pow((tl.y - bl.y), 2) );

    double maxWidth = max(widthA, widthB);
    double maxHeight = max(heightA, heightB);

    cv::Point2f src_vertices[4];
    src_vertices[0] = points[0];
    src_vertices[1] = points[1];
    src_vertices[2] = points[2];
    src_vertices[3] = points[3];

    cv::Point2f dst_vertices[4];
    dst_vertices[0] = cv::Point(0, 0);
    dst_vertices[1] = cv::Point(maxWidth - 1, 0);
    dst_vertices[2] = cv::Point(maxWidth - 1, maxHeight - 1);
    dst_vertices[3] = cv::Point(0, maxHeight - 1);
    cv::Mat warpMatrix = cv::getPerspectiveTransform(src_vertices, dst_vertices);

    //cout << "1: " << points[0] << ", 2: " << points[1] << ", 3: " << points[2] << ", 4: " << points[3] << endl;
    cv::Mat rotated;
    cv::warpPerspective(image, rotated, warpMatrix, cv::Size(maxWidth, maxHeight));


    return rotated;
}
