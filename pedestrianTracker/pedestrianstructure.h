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

#ifndef PEDESTRIANSTRUCTURE
#define PEDESTRIANSTRUCTURE

#include <opencv2/opencv.hpp>

typedef struct Pedestrian
{
    bool known;
    cv::Mat histogram;
    cv::Mat histogramImage;
    cv::Mat backProj;
    cv::Rect window;

}Pedestrian;

#endif // PEDESTRIANSTRUCTURE

