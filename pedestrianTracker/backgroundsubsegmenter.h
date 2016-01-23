/**
  * Projet de detection et tracking de pieton
  *
  * Thiriet Lucien
  * Counathe Kevin
  * Busy Maxime
  *
  * Filiere TDSI, departement GE, INSA de Lyon
  *
  **/

#ifndef BACKGROUNDSUBSEGMENTER_H
#define BACKGROUNDSUBSEGMENTER_H


#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>

class BackgroundSubSegmenter
{
private:

    cv::Mat mask;
    cv::Ptr<cv::BackgroundSubtractor> pMOG2;

public:

    BackgroundSubSegmenter();

    void detectPedestrians(cv::Mat sequence, std::vector<cv::Rect> &pedestrianDetected);
};

#endif // BACKGROUNDSUBSEGMENTER_H
