/**
  * Methode par différence de fond : OK
  * Methode de bloc matching : Pas encore implemente
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
    cv::Mat sequenceGrayDiff[nbTrames];
    cv::Mat sequenceBinary[nbTrames];
    cv::Mat sequenceMask[nbTrames];

    std::vector<std::vector<cv::Point> > contours; //detection des contours
    std::vector<cv::Vec4i> hierarchy;

    std::vector<std::vector<cv::Point> > contours_poly; // dessin des rectangles englobants
    std::vector<cv::Rect> boundRect;
    cv::Mat drawing[nbTrames];

    std::vector<cv::Mat> roi; //roi et histogrammes
    int histSize = 256;
    float range[] = {0,256};
    const float* histRange = {range};
    std::vector<std::vector<cv::Mat> > bgr_planes;

    std::vector<cv::MatND> histogramB;
    std::vector<cv::MatND> histogramG;
    std::vector<cv::MatND> histogramR;

    std::vector<cv::Mat> roiPreviousFrame;
    std::vector<cv::MatND> histogramBPreviousFrame;
    std::vector<cv::MatND> histogramGPreviousFrame;
    std::vector<cv::MatND> histogramRPreviousFrame;

    double distanceMoy;
    int ROI = 0;


    // soustracteur de fond
    cv::Ptr<cv::BackgroundSubtractor> pMOG2;
    pMOG2 = cv::createBackgroundSubtractorMOG2();

    //variables detection de points d'interets
    /*
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
    */

    //acquisition de la video
    for(int i =0;i<nbTrames;i++)
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

        /*
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
        */

    }
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
    //cv::namedWindow("Polygones", cv::WINDOW_AUTOSIZE);
    //cv::namedWindow("Histogramme", cv::WINDOW_AUTOSIZE);

    //traitement sur la video
    for(int i=0;i<nbTrames;i++)
    {

        cv::threshold(sequenceGrayDiff[i], sequenceBinary[i], threshold, 255, cv::THRESH_BINARY); //seuillage pour avoir notre masque

        cv::erode(sequenceBinary[i], sequenceMask[i], cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6,6)));   //erosion pour annuler le bruit du au vent
        cv::dilate(sequenceMask[i], sequenceMask[i], cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25,55))); // dilatation pour augmenter la taille des régions d'intérêt de notre masque
        cv::erode(sequenceMask[i], sequenceMask[i], cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,6)));   //erosion pour annuler le bruit du au vent


        ///Dessiner pour chaque canal de couleur histogramme
        /*
        for( int j = 1; j < histSize; j++ )
        {
            cv::line( histImage, cv::Point( bin_w*(j-1), hist_h - cvRound(b_hist[i].at<float>(j-1)) ), cv::Point( bin_w*(j), hist_h - cvRound(b_hist[i].at<float>(j)) ), cv::Scalar( 255, 0, 0), 2, 8, 0  );
            cv::line( histImage, cv::Point( bin_w*(j-1), hist_h - cvRound(g_hist[i].at<float>(j-1)) ), cv::Point( bin_w*(j), hist_h - cvRound(g_hist[i].at<float>(j)) ), cv::Scalar( 0, 255, 0), 2, 8, 0  );
            cv::line( histImage, cv::Point( bin_w*(j-1), hist_h - cvRound(r_hist[i].at<float>(j-1)) ), cv::Point( bin_w*(j), hist_h - cvRound(r_hist[i].at<float>(j)) ), cv::Scalar( 0, 0, 255), 2, 8, 0  );
        }
        */

        cv::findContours(sequenceMask[i], contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));

        contours_poly.resize(contours.size());
        boundRect.resize(contours.size());

        for( size_t j = 0; j < contours.size(); j++ )
        {
            cv::approxPolyDP(cv::Mat(contours[j]), contours_poly[j], 3, true);
            boundRect[j] = cv::boundingRect(cv::Mat(contours_poly[j]));
            roi.resize(boundRect.size());
            roi[j] = sequence[i](boundRect[j]);
        }

        drawing[i] = cv::Mat::zeros(sequenceMask[i].size(), CV_8UC3);

        //association des regions d'interet

        /*
        // si on a deja des ROI, on doit se baser dessus
        if(roiPreviousFrame.size() != 0 && boundRect.size() != 0)
        {
            std::cout<<"Frame "<<i<<" on avait des ROI on se base dessus, on a des ROI actuellement"<<std::endl;

            bgr_planes.resize(boundRect.size());
            histogramB.resize(boundRect.size());
            histogramG.resize(boundRect.size());
            histogramR.resize(boundRect.size());

            for(unsigned int j=0;j<boundRect.size();j++)
            {
                cv::split(sequence[i](boundRect[j]), bgr_planes[j]);

                cv::calcHist(&bgr_planes[j][0], 1, 0, cv::Mat(), histogramB[j], 1, &histSize, &histRange, true, false);
                cv::calcHist(&bgr_planes[j][1], 1, 0, cv::Mat(), histogramG[j], 1, &histSize, &histRange, true, false);
                cv::calcHist(&bgr_planes[j][2], 1, 0, cv::Mat(), histogramR[j], 1, &histSize, &histRange, true, false);

                cv::normalize(histogramB[j], histogramB[j], 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
                cv::normalize(histogramG[j], histogramG[j], 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
                cv::normalize(histogramR[j], histogramR[j], 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

                //std::cout<<cv::compareHist(histogramB[j], histogramB[0], CV_COMP_BHATTACHARYYA)<<std::endl;

                for(unsigned int k=0;k<roiPreviousFrame.size();k++)
                {
                    distanceMoy = (cv::compareHist(histogramB[j], histogramBPreviousFrame[k], CV_COMP_BHATTACHARYYA)+
                                   cv::compareHist(histogramB[j], histogramGPreviousFrame[k], CV_COMP_BHATTACHARYYA)+
                                   cv::compareHist(histogramB[j], histogramRPreviousFrame[k], CV_COMP_BHATTACHARYYA))/3;

                    if(distanceMoy < 0.5)
                    {
                        ROI = k;
                        break; // attention a ça
                    }
                    else if(roiPreviousFrame.size() < boundRect.size())
                    {
                        ROI = k+1;
                    }
                }

                roi.resize(boundRect.size());
                roi[ROI] = sequence[i](boundRect[j]);
            }
        }


        // dans le cas on on avait pas de ROI avant mais on a des ROI maintenant, il faut calculer les histogrammes

        else if(boundRect.size()!=0 && roiPreviousFrame.size() == 0)
        {
            std::cout<<"Frame "<<i<<" on avait pas de ROI et maintenant on en a"<<std::endl;

            bgr_planes.resize(boundRect.size());
            histogramB.resize(boundRect.size());
            histogramG.resize(boundRect.size());
            histogramR.resize(boundRect.size());

            for(unsigned int j=0;j<boundRect.size();j++)
            {
                roi.resize(boundRect.size());
                roi[j] = sequence[i](boundRect[j]); // les ROI sont placées dans un vecteur

                cv::split(sequence[i](boundRect[j]), bgr_planes[j]);

                cv::calcHist(&bgr_planes[j][0], 1, 0, cv::Mat(), histogramB[j], 1, &histSize, &histRange, true, false);
                cv::calcHist(&bgr_planes[j][1], 1, 0, cv::Mat(), histogramG[j], 1, &histSize, &histRange, true, false);
                cv::calcHist(&bgr_planes[j][2], 1, 0, cv::Mat(), histogramR[j], 1, &histSize, &histRange, true, false);

                cv::normalize(histogramB[j], histogramB[j], 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
                cv::normalize(histogramG[j], histogramG[j], 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
                cv::normalize(histogramR[j], histogramR[j], 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
            }
        }

        //dans le cas ou on a pas de ROI

        else if(boundRect.size() == 0)
        {
            std::cout<<"Frame "<<i<<" on a pas de ROI"<<std::endl;

            for(unsigned int j=0;j<boundRect.size();j++)
            {
                roi.resize(boundRect.size());
                roi[j] = sequence[i](boundRect[j]); // les ROI sont placées dans un vecteur
            }
        }
        */

        // dessins sur l'image finale
        for( size_t j = 0; j< contours.size(); j++ )
        {
            cv::drawContours(drawing[i], contours_poly, (int)j, cv::Scalar( 0, 0, 255), 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
            cv::rectangle(sequence[i], boundRect[j], cv::Scalar( 0, 0, 255), 2, 8, 0 );
            cv::putText(sequence[i], "ROI "+std::to_string(j), cv::Point(boundRect[j].x, boundRect[j].y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }

        //affichage de la video
        cv::imshow("Video", sequence[i]);
        //cv::imshow("Polygones", drawing[i]);
        //cv::imshow("Histogramme", histImage);

        //stocker les histogrammes et les ROIs dans des vecteurs, si il y a des ROIs

        roiPreviousFrame.clear();
        histogramBPreviousFrame.clear();
        histogramGPreviousFrame.clear();
        histogramRPreviousFrame.clear();

        if(roi.size()!=0)
        {
            roiPreviousFrame.resize(roi.size());
            histogramBPreviousFrame.resize(roi.size());
            histogramGPreviousFrame.resize(roi.size());
            histogramRPreviousFrame.resize(roi.size());

            roiPreviousFrame = roi;
            histogramBPreviousFrame = histogramB;
            histogramGPreviousFrame = histogramG;
            histogramRPreviousFrame = histogramR;
        }

        //on efface le vecteur contenant les points d'intérêts
        //corners.clear();
        contours_poly.clear();
        boundRect.clear();
        roi.clear();
        bgr_planes.clear();
        histogramB.clear();
        histogramG.clear();
        histogramR.clear();

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
