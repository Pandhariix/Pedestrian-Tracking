#include "watershedsegmenter.h"

WatershedSegmenter::WatershedSegmenter()
{

}

void WatershedSegmenter::setMarkers(const cv::Mat &markerImage)
{
    markerImage.convertTo(this->markers, CV_32S);
}

cv::Mat WatershedSegmenter::process(const cv::Mat &image)
{
    cv::watershed(image, this->markers);
    return this->markers;
}

cv::Mat WatershedSegmenter::getWatersheds()
{
    cv::Mat temp;
    this->markers.convertTo(temp, CV_8U, 255, 255);
    return temp;
}
