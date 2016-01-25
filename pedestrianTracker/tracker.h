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

#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <pedestrianstructure.h>
#include <blobpedestrianstructure.h>

class Tracker
{
private:

    int videoWidth;
    int videoHeight;

public:

    Tracker(int videoWidth, int videoHeight);

    void createFeatures(cv::Mat sequence, std::vector<cv::Rect> pedestriansSubDetected, std::vector<BlobPedestrian> &blobPedestrian);
    void trackFeatures(cv::Mat previousSequence, cv::Mat sequence, std::vector<BlobPedestrian> &blobPedestrian);
    void camshift(std::vector<Pedestrian> &pedestrian);
};

#endif // TRACKER_H
