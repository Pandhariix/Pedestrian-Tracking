/**
  * Methode par différence de fond
  * Distance de Battacharya
  * Methode de bloc matching
  *
  **/

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/background_segm.hpp>

#include <iostream>
#include <watershedsegmenter.h>

enum TrackingMethod {GOOD_FEATURES_TO_TRACK,
                     LUCAS_KANADE_TRACKING,
                     NOTHING_TO_TRACK};


//------------------------CHOIX-METHODE-DE-TRACKING----------------------------//

TrackingMethod chooseTrackingMethod(int nbRoi, int nbRoiPreviousFrame)
{
    if(nbRoi == 0)
    {
        return NOTHING_TO_TRACK;
    }
    else if(nbRoi > nbRoiPreviousFrame)
    {
        return GOOD_FEATURES_TO_TRACK;
    }
    else if(nbRoi == nbRoiPreviousFrame)
    {
        return LUCAS_KANADE_TRACKING; //TODO probleme initialisation
    }
    else
    {
        return LUCAS_KANADE_TRACKING;
    }
}


//------------------------CENTRE-DES-RECTANGLES--------------------------------//

std::vector<cv::Point2f> computeCentre(std::vector<cv::Rect> roi)
{
    std::vector<cv::Point2f> centers;
    centers.resize(roi.size());

    for(size_t i=0;i<roi.size();i++)
    {
        centers[i].x = roi[i].x+(roi[i].width/2);
        centers[i].y = roi[i].y+(roi[i].height/2);
    }

    return centers;
}



//------------------------BARYCENTRES-DES-RECTANGLES---------------------------//

void computeBarycentre()
{

}


//------------------------RATIO-RECTANGLES-------------------------------------//

std::vector<cv::Rect> rectangleRatio(std::vector<cv::Point2f> centres, std::vector<cv::Rect> rect)
{
    for(size_t i=0;i<rect.size();i++)
    {
        if((rect[i].width/rect[i].height) > 0.33)
        {
            rect[i].width = 0.33*rect[i].height;
            rect[i].x
        }
    }
}



//------------------------MAIN-------------------------------------------------//


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
    int threshold = 150;

    cv::Mat sequence[nbTrames];     //the sequence of images for the video
    cv::Mat sequenceGray[nbTrames];
    cv::Mat sequenceGrayDiff[nbTrames];
    cv::Mat sequenceBinary[nbTrames];
    cv::Mat sequenceMask[nbTrames];

    std::vector<std::vector<cv::Point> > contours; //detection des contours
    std::vector<cv::Vec4i> hierarchy;

    std::vector<std::vector<cv::Point> > contours_poly; // dessin des rectangles englobants
    std::vector<cv::Rect> boundRect;

    cv::Mat maskTemp;
    cv::Mat roiMask;
    std::vector<cv::Mat> mask;

    int nbRoiPreviousFrame = 0;

    TrackingMethod choiceTracking = NOTHING_TO_TRACK;

    //Lucas Kanade recalage
    cv::Mat previousSequenceGray;
    std::vector<uchar> status;
    std::vector<float> err;

    //dessin des rectangles
    std::vector<cv::Rect> rect;
    std::vector<cv::Rect> previousRect;

    // soustracteur de fond
    cv::Ptr<cv::BackgroundSubtractor> pMOG2;
    pMOG2 = cv::createBackgroundSubtractorMOG2();

    //variables detection de points d'interets
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<std::vector<cv::Point2f>> previousCorners;
    int maxCorners = 50;
    double qualityLevel = 0.01;
    double minDistance = 2.;

    int blockSize = 3;
    bool useHarrisDetector = false;
    double kdef = 0.04;


    //variables centre rectangle
    std::vector<cv::Point2f> centres;



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
        pMOG2->apply(sequence[i],sequenceGrayDiff[i]);
    }

    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);

    //traitement sur la video
    for(int i=0;i<nbTrames;i++)
    {
        cv::cvtColor(sequence[i], sequenceGray[i], CV_BGR2GRAY);


        cv::threshold(sequenceGrayDiff[i], sequenceBinary[i], threshold, 255, cv::THRESH_BINARY); //seuillage pour avoir notre masque

        cv::erode(sequenceBinary[i], sequenceMask[i], cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6,6)));   //erosion pour annuler le bruit du au vent
        cv::dilate(sequenceMask[i], sequenceMask[i], cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25,55))); // dilatation pour augmenter la taille des régions d'intérêt de notre masque
        cv::erode(sequenceMask[i], sequenceMask[i], cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,6)));   //erosion pour annuler le bruit du au vent

        sequenceMask[i].copyTo(maskTemp);

        cv::findContours(sequenceMask[i], contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));

        contours_poly.resize(contours.size());
        boundRect.resize(contours.size());


        for(size_t j=0;j<contours.size();j++)
        {
            cv::approxPolyDP(cv::Mat(contours[j]), contours_poly[j], 3, true);
            boundRect[j] = cv::boundingRect(cv::Mat(contours_poly[j]));
            mask.resize(boundRect.size());

            mask[j] = cv::Mat::zeros(maskTemp.size(), CV_8UC1);
            roiMask = mask[j](boundRect[j]);
            maskTemp(boundRect[j]).copyTo(roiMask);
        }

        choiceTracking = chooseTrackingMethod(boundRect.size(),nbRoiPreviousFrame);

        if(choiceTracking == GOOD_FEATURES_TO_TRACK)
        {
            //detection des points d'interet pour chaque bounding box
            corners.resize(boundRect.size());

            for(size_t j=0;j<boundRect.size();j++)
            {
                cv::goodFeaturesToTrack(sequenceGray[i], corners[j], maxCorners, qualityLevel, minDistance, mask[j], blockSize, useHarrisDetector, kdef);
            }
        }
        else if(choiceTracking == LUCAS_KANADE_TRACKING)
        {
            corners.resize(previousCorners.size());

            for(size_t j=0;j<boundRect.size();j++)
            {
                cv::calcOpticalFlowPyrLK(previousSequenceGray, sequenceGray[i], previousCorners[j], corners[j], status, err);
            }
        }


        //------------------------CALCUL-DES-CENTRES---DEFINITION-DES-RECTANGLES-------//

        if(choiceTracking != NOTHING_TO_TRACK)
        {
            rect.resize(corners.size());

            for( size_t j = 0; j< corners.size(); j++ )
            {
                rect[j] = cv::boundingRect(corners[j]);
            }

            centres.resize(corners.size());
            centres = computeCentre(rect);
        }

        /*
        if(choiceTracking != NOTHING_TO_TRACK)
        {
            //placement des points d'interêts sur l'image POUR LE DEBUG
            for(size_t j=0;j<corners.size();j++)
            {
                for(size_t k=0;k<corners[j].size();k++)
                {
                    if(j%3 == 0){
                        cv::circle(sequence[i], corners[j][k], 1, cv::Scalar(0,0,255),-1);
                    }
                    else if(j%2 == 0){
                        cv::circle(sequence[i], corners[j][k], 1, cv::Scalar(0,255,0),-1);
                    }
                    else{
                        cv::circle(sequence[i], corners[j][k], 1, cv::Scalar(255,0,0),-1);
                    }
                }
            }
        }
        */


        // dessins sur l'image finale
        if(choiceTracking != NOTHING_TO_TRACK)
        {
            for( size_t j = 0; j< corners.size(); j++ )
            {
                cv::rectangle(sequence[i], rect[j], cv::Scalar( 0, 0, 255), 2, 8, 0 );
                cv::putText(sequence[i], "ROI", cv::Point(rect[j].x, rect[j].y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                cv::circle(sequence[i], centres[j], 1, cv::Scalar(0,255,0), -1, 24);
            }
        }


        //affichage de la video
        cv::imshow("Video", sequence[i]);


        //description de la frame pour la frame suivante
        previousSequenceGray.release();

        nbRoiPreviousFrame = contours.size();
        previousSequenceGray = sequenceGray[i];

        previousCorners.clear();
        previousCorners.resize(corners.size());
        previousCorners = corners;

        //nettoyage des variables
        contours.clear();
        hierarchy.clear();
        contours_poly.clear();
        boundRect.clear();
        maskTemp.release();
        roiMask.release();
        mask.clear();
        status.clear();
        err.clear();
        corners.clear();
        rect.clear();
        previousRect.clear();
        centres.clear();

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
