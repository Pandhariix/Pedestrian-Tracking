/**
  * HOG Detection
  *
  **/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

std::vector<cv::Rect> hogDetection(cv::Mat sequence, cv::HOGDescriptor hog)
{
    std::vector<cv::Rect> found;
    std::vector<cv::Rect> found_filtered;

    hog.detectMultiScale(sequence, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);

    size_t k, j;

    for (k=0; k<found.size(); k++)
    {
        cv::Rect r = found[k];

        for (j=0; j<found.size(); j++)
            if (j!=k && (r & found[j]) == r)
                break;

        if (j== found.size())
            found_filtered.push_back(r);
    }
    return found_filtered;
}

std::vector<std::vector<cv::Point2f>> featuresDetection(cv::Mat sequenceGray, std::vector<cv::Rect> found_filtered)
{
    std::vector<std::vector<cv::Point2f>> corners;
    int maxCorners = 25;
    double qualityLevel = 0.01;
    double minDistance = 1.;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double kdef = 0.04;

    corners.resize(found_filtered.size());

    for(size_t j=0;j<found_filtered.size();j++)
    {
        cv::goodFeaturesToTrack(sequenceGray(found_filtered[j]), corners[j], maxCorners, qualityLevel, minDistance, cv::noArray(), blockSize, useHarrisDetector, kdef);

        for(size_t k=0;k<corners[j].size();k++)
        {
            corners[j][k].x += found_filtered[j].x;
            corners[j][k].y += found_filtered[j].y;
        }
    }
    return corners;
}

std::vector<std::vector<cv::Point2f>> lucasKanadeTracking(cv::Mat previousSequenceGray, cv::Mat sequenceGray, std::vector<std::vector<cv::Point2f>> corners)
{
    std::vector<std::vector<cv::Point2f>> newCorners;
    std::vector<uchar> status;
    std::vector<float> err;

    newCorners.resize(corners.size());

    for(size_t j=0;j<corners.size();j++)
    {
        cv::calcOpticalFlowPyrLK(previousSequenceGray, sequenceGray, corners[j], newCorners[j], status, err);
    }
    return newCorners;
}


int main(int argc, char *argv[])
{

    //test pour savoir si l'utilisateur a renseigne un parametre
    if(argc == 1)
    {
        std::cout<<"Veuillez rentrer un parametre"<<std::endl;
        std::exit(EXIT_FAILURE);
    }

    //variables images et masque

    std::string inputFileName(argv[1]);

    int nbTrames = 501;

    cv::Mat sequence[nbTrames];     //the sequence of images for the video
    cv::Mat sequenceGray[nbTrames];
    cv::Mat previousSequenceGray;

    int nbPedestrians = 0;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    std::vector<cv::Rect> detectedPedestrian;

    std::vector<std::vector<cv::Point2f>> featuresDetected;
    std::vector<std::vector<cv::Point2f>> previousFeaturesDetected;

    //acquisition de la video
    for(int i=0;i<nbTrames;i++)
    {
        std::stringstream nameTrame;
        if(i<10)
        {
            nameTrame << inputFileName << "_000" << i << ".jpeg";
        }
        else if(i<100)
        {
            nameTrame << inputFileName << "_00" << i << ".jpeg";
        }
        else
        {
            nameTrame << inputFileName << "_0" << i << ".jpeg";
        }

        std::cout<<nameTrame.str()<<std::endl;

        sequence[i] = cv::imread(nameTrame.str());
    }

    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);


    //traitement sur la video
    for(int i=0;i<nbTrames;i++)
    {
        cv::cvtColor(sequence[i], sequenceGray[i], CV_BGR2GRAY);

        if(i>0)
            previousSequenceGray = sequenceGray[i-1];
        else
            previousSequenceGray = sequenceGray[i];


        if(i%15 == 0)
        {
            detectedPedestrian = hogDetection(sequence[i], hog);
            nbPedestrians = detectedPedestrian.size();

            if(nbPedestrians != 0)
            {
                featuresDetected = featuresDetection(sequenceGray[i], detectedPedestrian);
                previousFeaturesDetected.resize(featuresDetected.size());
                previousFeaturesDetected = featuresDetected;
            }
        }
        else if(previousFeaturesDetected.size() != 0)
        {
            featuresDetected = lucasKanadeTracking(previousSequenceGray, sequenceGray[i], previousFeaturesDetected);

            previousFeaturesDetected.clear();
            previousFeaturesDetected.resize(featuresDetected.size());
            previousFeaturesDetected = featuresDetected;
        }

        /*
        for(size_t j=0;j<featuresDetected.size();j++)
        {
            for(size_t k=0;k<featuresDetected[j].size();k++)
            {
                cv::circle(sequence[i], featuresDetected[j][k], 1, cv::Scalar(0,0,255),-1);
            }
        }
        */

        for(size_t j=0;j<featuresDetected.size();j++)
        {
            cv::rectangle(sequence[i], cv::boundingRect(featuresDetected[j]), cv::Scalar( 0, 0, 255), 2, 8, 0 );
        }

        //affichage de la video
        cv::imshow("Video", sequence[i]);

        //clear des variables
        detectedPedestrian.clear();
        featuresDetected.clear();

        previousSequenceGray.release();

        //condition arret
        if (cv::waitKey(66) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            return 0;
        }
    }

    cv::waitKey(0);
    return 0;
}
