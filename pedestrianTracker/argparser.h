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

#ifndef ARGPARSER_H
#define ARGPARSER_H

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

enum formatVideo{SEQUENCE_IMAGE,
                 SEQUENCE_VIDEO,
                 UNDEFINED_FORMAT};

enum trackingAlgo{BLOB_TRACKING,
                  CAMSHIFT_TRACKING,
                  UNDEFINED_ALGORITHM};

class ArgParser
{
private:

    std::vector<std::string> args;
    formatVideo formatType;
    trackingAlgo algorithm;

public:

    ArgParser();
    ArgParser(const int argc, std::string format, std::string file, std::string tracking_algorithm);

    void detectFormat();
    void detectTrackingAlgorithm();
    void extractVideo(std::vector<cv::Mat> &sequence, int &nbTrames, double &fps); //sequence image : nbTrames et fps a remplir
    trackingAlgo selectedAlgorithm();
};

#endif // ARGPARSER_H
