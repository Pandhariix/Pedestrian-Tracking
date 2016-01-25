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

#ifndef PEDESTRIANBUILDER_H
#define PEDESTRIANBUILDER_H

#include <opencv2/opencv.hpp>
#include <pedestrianstructure.h>

class PedestrianBuilder
{
private:

    int videoWidth;
    int videoHeight;

public:

    PedestrianBuilder(int videoWidth, int videoHeight);

    void detectNewPedestrian(std::vector<Pedestrian> &pedestrian, std::vector<cv::Rect> pedestrianSubDetected);
    void buildPedestrian(std::vector<Pedestrian> &pedestrian, cv::Mat sequence);
};

#endif // PEDESTRIANBUILDER_H
