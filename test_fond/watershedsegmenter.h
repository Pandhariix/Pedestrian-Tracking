#ifndef WATERSHEDSEGMENTER_H
#define WATERSHEDSEGMENTER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>


class WatershedSegmenter
{
private:

    cv::Mat markers;

public:

    WatershedSegmenter();

    void setMarkers(const cv::Mat &markerImage);
    cv::Mat process(const cv::Mat &image);
    cv::Mat getWatersheds();

};

#endif // WATERSHEDSEGMENTER_H
