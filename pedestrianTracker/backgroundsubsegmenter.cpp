#include "backgroundsubsegmenter.h"

BackgroundSubSegmenter::BackgroundSubSegmenter()
{
    this->pMOG2 = cv::createBackgroundSubtractorMOG2();
}


void BackgroundSubSegmenter::detectPedestrians(cv::Mat sequence, std::vector<cv::Rect> &pedestrianDetected)
{
    int threshold = 150;
    cv::Mat maskCopy;
    cv::Mat sequenceGrayDiff;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<std::vector<cv::Point>> contours_poly;

    pedestrianDetected.clear();

    this->pMOG2->apply(sequence,sequenceGrayDiff);

    cv::threshold(sequenceGrayDiff, this->mask, threshold, 255, cv::THRESH_BINARY);
    cv::erode(this->mask, this->mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6,6)));
    cv::dilate(this->mask, this->mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25,55)));
    cv::erode(this->mask, this->mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,6)));

    this->mask.copyTo(maskCopy);
    cv::findContours(maskCopy, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));

    contours_poly.resize(contours.size());
    pedestrianDetected.resize(contours.size());

    for(size_t j=0;j<contours.size();j++)
    {
        cv::approxPolyDP(cv::Mat(contours[j]), contours_poly[j], 3, true);
        pedestrianDetected[j] = cv::boundingRect(cv::Mat(contours_poly[j]));
    }
}
