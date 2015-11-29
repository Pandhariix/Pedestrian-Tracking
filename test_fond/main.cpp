/**
  * Methode par différence de fond : OK
  * Methode de bloc matching : Pas encore implemente
  *
  **/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <watershedsegmenter.h>

int main()
{

    ///TEST AVEC VIDEO DATA

    //variables images et masque
    int nbTrames = 501;
    int threshold = 30;
    cv::Mat sequence[nbTrames];     //the sequence of images for the video
    cv::Mat sequenceGray[nbTrames];
    cv::Mat sequenceGrayDiff[nbTrames];
    cv::Mat sequenceBinary[nbTrames];
    cv::Mat sequenceBinaryErodeDilate[nbTrames];
    cv::Mat sequenceMask[nbTrames];
    cv::Mat sequenceEdges[nbTrames];
    cv::Mat sequenceMarkers[nbTrames];

    WatershedSegmenter segmenter;

    //variables detection de points d'interets
    std::vector <cv::Point2f> corners;
    int maxCorners = 100;
    double qualityLevel = 0.01;
    double minDistance = 20.;

    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    //variables pour histogramme
    int histSize = 256;
    float range[] = {0,256};
    const float* histRange = {range};
    std::vector<cv::Mat> bgr_planes[nbTrames];

    bool uniform = true;
    bool accumulate = false;

    cv::Mat b_hist[nbTrames];
    cv::Mat g_hist[nbTrames];
    cv::Mat r_hist[nbTrames];

    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double) hist_w/histSize);

    cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );


    //acquisition de la video
    for(int i =0;i<nbTrames;i++)
    {
        std::stringstream nameTrame;
        if(i<10)
        {
            nameTrame << "Data/tracking_000" << i << ".jpeg";
        }
        else if(i<100)
        {
            nameTrame << "Data/tracking_00" << i << ".jpeg";
        }
        else
        {
            nameTrame << "Data/tracking_0" << i << ".jpeg";
        }

        std::cout<<nameTrame.str()<<std::endl;

        sequence[i] = cv::imread(nameTrame.str());

        //split les composantes de l'image pour le calcul de l'histogramme
        cv::split(sequence[i], bgr_planes[i]);

        // Compute the histograms:
        cv::calcHist( &bgr_planes[i][0], 1, 0, cv::Mat(), b_hist[i], 1, &histSize, &histRange, uniform, accumulate );
        cv::calcHist( &bgr_planes[i][1], 1, 0, cv::Mat(), g_hist[i], 1, &histSize, &histRange, uniform, accumulate );
        cv::calcHist( &bgr_planes[i][2], 1, 0, cv::Mat(), r_hist[i], 1, &histSize, &histRange, uniform, accumulate );

        //normalise les histogrammes
        cv::normalize(b_hist[i], b_hist[i], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(g_hist[i], g_hist[i], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
        cv::normalize(r_hist[i], r_hist[i], 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    }

    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
    //cv::namedWindow("Histogramme", cv::WINDOW_AUTOSIZE);

    //traitement sur la video
    for(int i=0;i<nbTrames;i++)
    {

        cv::cvtColor(sequence[i], sequenceGray[i], CV_BGR2GRAY); //passage en gris
        cv::absdiff(sequenceGray[0], sequenceGray[i], sequenceGrayDiff[i]); // différence des images
        cv::threshold(sequenceGrayDiff[i], sequenceBinary[i], threshold, 255, 1); //seuillage pour avoir notre masque
        cv::bitwise_not(sequenceBinary[i], sequenceBinary[i]); //on met nos zones d'intérêt en blanc

        cv::erode(sequenceBinary[i], sequenceBinaryErodeDilate[i], cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6,6)));   //erosion pour annuler le bruit du au vent
        cv::dilate(sequenceBinaryErodeDilate[i], sequenceBinaryErodeDilate[i], cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20,20))); // dilatation pour augmenter la taille des régions d'intérêt de notre masque
        cv::min(sequenceBinaryErodeDilate[i], sequenceBinary[i], sequenceMask[i]);

        ///Good features to track

        //detection des points d'interet
        cv::goodFeaturesToTrack( sequenceGray[i], corners, maxCorners, qualityLevel, minDistance, sequenceMask[i], blockSize, useHarrisDetector, k );

        //placement des points d'interêts sur l'image POUR LE DEBUG
        for(size_t j = 0; j < corners.size(); j++)
        {
            cv::circle(sequence[i], corners[j], 1, cv::Scalar(0,255,0),-1);
        }


        ///Dessiner pour chaque canal de couleur histogramme

        for( int j = 1; j < histSize; j++ )
        {
            cv::line( histImage, cv::Point( bin_w*(j-1), hist_h - cvRound(b_hist[i].at<float>(j-1)) ), cv::Point( bin_w*(j), hist_h - cvRound(b_hist[i].at<float>(j)) ), cv::Scalar( 255, 0, 0), 2, 8, 0  );
            cv::line( histImage, cv::Point( bin_w*(j-1), hist_h - cvRound(g_hist[i].at<float>(j-1)) ), cv::Point( bin_w*(j), hist_h - cvRound(g_hist[i].at<float>(j)) ), cv::Scalar( 0, 255, 0), 2, 8, 0  );
            cv::line( histImage, cv::Point( bin_w*(j-1), hist_h - cvRound(r_hist[i].at<float>(j-1)) ), cv::Point( bin_w*(j), hist_h - cvRound(r_hist[i].at<float>(j)) ), cv::Scalar( 0, 0, 255), 2, 8, 0  );
        }


        /*
        if(i!=0)
        {
            cv::Canny(sequenceMask[i], sequenceEdges[i], 30, 100, 3, false);
            cv::connectedComponents(sequenceMask[i], sequenceMarkers[i], 8, 4);
        }
        */


        //affichage de la video
        cv::imshow("Video", sequence[i]);

        //cv::imshow("Histogramme", histImage);

        //on efface le vecteur contenant les points d'intérêts
        corners.clear();

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
