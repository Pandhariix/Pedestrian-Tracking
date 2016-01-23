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
                 UNDEFINED};

class ArgParser
{
private:

    std::vector<std::string> args;
    formatVideo formatType;

public:

    ArgParser();
    ArgParser(const int argc, std::string format, std::string file);

    void detectFormat();
    void extractVideo(std::vector<cv::Mat> &sequence, int &nbTrames, double &fps); //sequence image : nbTrames et fps a remplir
};

#endif // ARGPARSER_H
