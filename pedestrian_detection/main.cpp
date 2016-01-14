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
#include "opencv2/features2d/features2d.hpp"

#include <iostream>



//------------------ENUMERATIONS-------------------------------------------//


enum choiceAlgo{HOG_TEMPLATE_TRACKING,
                HOG_GOODFEATURESTOTRACK_LK,
                OPT_FLOW_FARNEBACK,
                CAMSHIFT_KALMAN_FILTER};

enum formatVideo{SEQUENCE_IMAGE,
                 VIDEO,
                 NOT_DEFINED};




//------------------METHODES-PARSERS---------------------------------------//


/// Detection de l'algorithme utilisé

choiceAlgo detectAlgo(std::string argument)
{
    if(argument.compare("template_tracking") == 0)
        return HOG_TEMPLATE_TRACKING;

    else if(argument.compare("LK_tracking") == 0)
        return HOG_GOODFEATURESTOTRACK_LK;

    else if(argument.compare("farneback") == 0)
        return OPT_FLOW_FARNEBACK;

    else if(argument.compare("camshift_kalman") == 0)
        return CAMSHIFT_KALMAN_FILTER;

    else
        return HOG_GOODFEATURESTOTRACK_LK;

}


/// Detection du format video et extraction des données

formatVideo detectFormat(std::string argument)
{
    if(argument.compare("image") == 0)
        return SEQUENCE_IMAGE;

    else if(argument.compare("video") == 0)
        return VIDEO;

    else
        return NOT_DEFINED;
}



std::vector<cv::Mat> extractVideoData(formatVideo format, std::string filePathName, int nbTrames)
{
    std::vector<cv::Mat> sequence;

    if(format == SEQUENCE_IMAGE)
    {
        sequence.resize(nbTrames);

        for(int i=0;i<nbTrames;i++)
        {
            std::stringstream nameTrame;
            if(i<10)
            {
                nameTrame << filePathName << "_000" << i << ".jpeg";
            }
            else if(i<100)
            {
                nameTrame << filePathName << "_00" << i << ".jpeg";
            }
            else
            {
                nameTrame << filePathName << "_0" << i << ".jpeg";
            }

            std::cout<<nameTrame.str()<<std::endl;

            sequence[i] = cv::imread(nameTrame.str());
        }
        return sequence;
    }

    else if(format == VIDEO)
    {
        cv::VideoCapture capture;
        capture.open(filePathName);

        if(capture.isOpened())
        {
            double fps = capture.get(CV_CAP_PROP_FPS);
            int delay = 1000/fps;
            int nbFrames = 0;

            while(true)
            {
                if(!capture.read(sequence[nbFrames]))
                    break;

                if(cv::waitKey(delay)>=0)
                    break;
            }
            capture.release();
        }

        else
        {
            std::cout<<"Impossible de lire la video : "<<filePathName<<std::endl;
        }

        return sequence;
    }

    else
    {
        std::cout<<"Argument incorrect, veuillez entrer 'image' ou 'video'"<<std::endl;
        return sequence;
    }
}






//------------------METHODES-HOG-------------------------------------------//


/// Detection via le HOG detector de opencv des piétons

std::vector<cv::Rect> hogDetection(cv::Mat sequence, cv::HOGDescriptor hog)
{
    std::vector<cv::Rect> found;

    hog.detectMultiScale(sequence, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);

    return found;
}



//------------------METHODES-CORNERS-ET-LK-TRACKING------------------------//


///Corners detection (detection des corners via good features to track)

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




/// Tracking des points d'interêts determinés avec méthode de Lucas Kanade

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




//------------------METHODES-TEMPLATES-------------------------------------//


/// Tracking avec méthode des templates

std::vector<cv::Rect> templateTracking(cv::Mat sequence, std::vector<cv::Rect> foundFilteredTemplate, int matchingMethod)
{
    cv::Mat result;
    std::vector<cv::Rect> roi;
    int method;
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::Point matchLoc;

    roi.resize(foundFilteredTemplate.size());

    switch(matchingMethod){

    case 0:
        method = CV_TM_SQDIFF;
        break;

    case 1:
        method = CV_TM_SQDIFF_NORMED;
        break;

    case 2:
        method = CV_TM_CCORR;
        break;

    case 3:
        method = CV_TM_CCORR_NORMED;
        break;

    case 4:
        method = CV_TM_CCOEFF;
        break;

    case 5:
        method = CV_TM_CCOEFF_NORMED;
        break;

    default:
        method = CV_TM_CCORR_NORMED;
        break;

    }

    for(size_t i=0;i<foundFilteredTemplate.size();i++)
    {
        cv::matchTemplate(sequence, sequence(foundFilteredTemplate[i]), result, method);
        cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

        if(method == CV_TM_SQDIFF || method == CV_TM_SQDIFF_NORMED)
            matchLoc = minLoc;
        else
            matchLoc = maxLoc;

        roi[i].x = matchLoc.x;
        roi[i].y = matchLoc.y;
        roi[i].width = foundFilteredTemplate[i].width;
        roi[i].height = foundFilteredTemplate[i].height;

        result.release();
    }

    return roi;

}




//------------------METHODES-OPTICAL-FLOW-FARNEBACK------------------------//


/// Dessin d'un optical flow farneback

void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step, const cv::Scalar& color)
{
        for(int y = 0; y < cflowmap.rows; y += step)
            for(int x = 0; x < cflowmap.cols; x += step)
            {
                const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
                cv::line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
                cv::circle(cflowmap, cv::Point(x,y), 2, color, -1);
            }
}






//------------------METHODES-CAMSHIFT-ET-KALMAN-FILTER---------------------//


/// Generation de la back projection d'un histogramme, obtenu a partir d'une image

std::vector<cv::MatND> computeProbImage(cv::Mat image, std::vector<cv::Rect> rectRoi, std::vector<cv::Mat> &hist, std::vector<bool> &detected)
{
    int smin = 30;
    int vmin = 10;
    int vmax = 256;
    cv::Mat mask;
    cv::Mat hsv;
    cv::Mat hue;
    std::vector<cv::MatND> backProj;
    int channels[] = {0,0};
    int hbins = 30;                                   // Quantize the hue to 30 levels
    //int sbins = 32;                                 // and the saturation to 32 levels
    int histSize = MAX( hbins, 2 );
    //int histSizes[] = {hbins, sbins};
    float hue_range[] = { 0, 180 };                   // hue varies from 0 to 179, see cvtColor
    //float sat_range[] = { 0, 256 };                 // saturation varies from 0 (black-gray-white) to
    const float* range = { hue_range };               // 255 (pure spectrum color)
    //const float* ranges = { hue_range, sat_range };
    //double maxVal=0;

    backProj.resize(rectRoi.size());
    hist.resize(rectRoi.size());

    cv::cvtColor(image, hsv, CV_BGR2HSV);
    hue.create(hsv.size(), hsv.depth());
    cv::mixChannels(&hsv, 1, &hue, 1, channels, 1);
    cv::inRange(hsv, cv::Scalar(0, smin, MIN(vmin,vmax)), cv::Scalar(180, 256, MAX(vmin, vmax)), mask);

    for(size_t i=0;i<rectRoi.size();i++)
    {
        if(!detected[i])
        {
            cv::Mat roi(hue, rectRoi[i]);
            cv::Mat maskroi(mask, rectRoi[i]);

            cv::calcHist(&roi, 1, 0, maskroi, hist[i], 1, &histSize, &range, true, false);
            cv::normalize(hist[i], hist[i], 0, 255, cv::NORM_MINMAX);

            detected[i] = true;

            roi.release();
            maskroi.release();
        }

        cv::calcBackProject(&hue, 1, 0, hist[i], backProj[i], &range);
        backProj[i] &= mask;
    }

    return backProj;
}




/// Comparaisons de roi, pour determiner si il y a deja eu detection

void refineROI(std::vector<cv::Rect> &roiRefined, std::vector<bool> &detected, std::vector<cv::Rect> roiHog)
{
    if(roiRefined.size() != 0)
    {
        for(size_t i=0;i<roiRefined.size();i++)
        {
            for(size_t j=0;j<roiHog.size();j++)
            {
                const cv::Point p1(roiHog[j].x, roiHog[j].y+roiHog[j].height);
                const cv::Point p2(roiHog[j].x+roiHog[j].width, roiHog[j].y+roiHog[j].height);
                const cv::Point p3(roiHog[j].x+roiHog[j].width, roiHog[j].y);
                const cv::Point p4(roiHog[j].x, roiHog[j].y);

                if(!roiRefined[i].contains(p1) ||
                   !roiRefined[i].contains(p2) ||
                   !roiRefined[i].contains(p3) ||
                   !roiRefined[i].contains(p4))
                {
                    roiRefined.push_back(roiHog[j]);
                    detected.push_back(false);
                }
            }
        }
    }
    else
    {
        for(size_t i=0;i<roiHog.size();i++)
        {
            roiRefined.push_back(roiHog[i]);
            detected.push_back(false);
        }
    }
}






//------------------MAIN---------------------------------------------------//

int main(int argc, char *argv[])
{

    //test pour savoir si l'utilisateur a renseigne un parametre
    if(argc <= 3)
    {
        std::cout<<"---------------------------------------"<<std::endl<<
                   "Veuillez rentrer la methode choisie :  "<<std::endl<<
                   "- template_tracking"                    <<std::endl<<
                   "- LK_tracking"                          <<std::endl<<
                   "- farneback"                            <<std::endl<<
                   "- camshift_kalman"           <<std::endl<<std::endl<<
                   "Le type de format : "                   <<std::endl<<
                   "- video"                                <<std::endl<<
                   "- image"                     <<std::endl<<std::endl<<
                   "Le nom du fichier d'input"              <<std::endl<<
                   "---------------------------------------"<<std::endl;
        std::exit(EXIT_FAILURE);
    }



    //------------------VARIABLES----------------------------------------------//


    //Variables communes

    choiceAlgo algo;
    formatVideo format;
    std::string inputFileName(argv[3]);
    int nbTrames = 501;

    std::vector<cv::Mat> sequence;
    std::vector<cv::Mat> sequenceGray;

    cv::Mat previousSequenceGray;

    int nbPedestrians = 0;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    std::vector<cv::Rect> detectedPedestrian;


    // HOG + Good feature to track + LK
    std::vector<std::vector<cv::Point2f>> featuresDetected;
    std::vector<std::vector<cv::Point2f>> previousFeaturesDetected;



    // HOG + Template tracking
    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect> previousBoxes;



    // Optical flow farneback
    cv::Mat flow;
    cv::Mat imGray;
    cv::Mat imGrayPrev;



    //camshift and kalman filter
    std::vector<cv::MatND> backProj;
    std::vector<cv::Rect> roiHogDetected;
    std::vector<cv::Rect> roiCamShift;
    std::vector<bool> detected;
    std::vector<cv::Mat> hist;
    std::vector<cv::RotatedRect> rectCamShift;
    cv::Point2f rect_points[4];



    //acquisition de la video
    algo = detectAlgo(std::string(argv[1]));
    format = detectFormat(std::string(argv[2]));





    //------------------VIDEO--------------------------------------------------//

    if(format == SEQUENCE_IMAGE)
        sequence.resize(nbTrames);
    else if(format == VIDEO)
        std::cout<<"video";

    sequence = extractVideoData(format, inputFileName, nbTrames);
    sequenceGray.resize(sequence.size());

    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);





    //------------------TRAITEMENT-VIDEO---------------------------------------//

    for(int i=0;i<nbTrames;i++)
    {
        cv::cvtColor(sequence[i], sequenceGray[i], CV_BGR2GRAY);

        if(i>0)
            previousSequenceGray = sequenceGray[i-1];
        else
            previousSequenceGray = sequenceGray[i];




        ///------------------HOG + Good Features to track + LK-----------------//

        if(algo == HOG_GOODFEATURESTOTRACK_LK)
        {
            if(i%20 == 0)
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


            //--------Representation--------------------

            /*
            cv::Scalar myColor;

            for(size_t j=0;j<featuresDetected.size();j++)
            {
                if(j%3 == 0)
                    myColor = cv::Scalar(0,0,cv::RNG().uniform(200,255));

                else if(j%2 == 0)
                    myColor = cv::Scalar(0,cv::RNG().uniform(200,255),0);

                else
                    myColor = cv::Scalar(cv::RNG().uniform(200,255),0,0);

                for(size_t k=0;k<featuresDetected[j].size();k++)
                {
                    cv::circle(sequence[i], featuresDetected[j][k], 1, myColor,-1);
                }
            }
            */


            for(size_t j=0;j<featuresDetected.size();j++)
            {
                cv::rectangle(sequence[i], cv::boundingRect(featuresDetected[j]), cv::Scalar( 0, 0, 255), 2, 8, 0 );
            }



            //affichage de la video
            cv::imshow("Video", sequence[i]);
        }





        ///------------------HOG + Template Tracking---------------------------//

        else if(algo == HOG_TEMPLATE_TRACKING)
        {
            if(i%20 == 0)
            {
                detectedPedestrian = hogDetection(sequence[i], hog);
                nbPedestrians = detectedPedestrian.size();

                if(nbPedestrians != 0)
                {
                    boxes = templateTracking(sequence[i], detectedPedestrian, CV_TM_CCORR_NORMED);
                    previousBoxes.resize(boxes.size());
                    previousBoxes = boxes;
                }
            }
            else if(previousBoxes.size() != 0)
            {
                boxes = templateTracking(sequence[i], previousBoxes, CV_TM_CCORR_NORMED);

                previousBoxes.clear();
                previousBoxes.resize(boxes.size());
                previousBoxes = boxes;
            }

            //--------Representation--------------------

            for(size_t j=0;j<boxes.size();j++)
            {
                cv::rectangle(sequence[i], boxes[j], cv::Scalar( 0, 0, 255), 2, 8, 0 );
            }


            //affichage de la video
            cv::imshow("Video", sequence[i]);
        }






        ///------------------HOG + Optical Flow Farneback----------------------//

        else if(algo == OPT_FLOW_FARNEBACK)
        {
            if(i!=0)
            {
                flow = cv::Mat::zeros(sequence[i].size(), CV_32FC2);
                cv::cvtColor(sequence[i], imGray, CV_BGR2GRAY);
                cv::cvtColor(sequence[i-1], imGrayPrev, CV_BGR2GRAY);

                cv::calcOpticalFlowFarneback(imGrayPrev, imGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);


                //-----------------Representation------------------------------//

                drawOptFlowMap(flow, imGrayPrev, 16, CV_RGB(0, 255, 0)); //dessin test

                //affichage de la video
                cv::imshow("Video", imGrayPrev);
            }
        }





        ///--------------HOG+Camshift + Kalman Filter--------------------------//

        else if(algo == CAMSHIFT_KALMAN_FILTER)
        {

            //camshift
            if(i%20 == 0 && roiCamShift.size() == 0)
            {
                roiHogDetected = hogDetection(sequence[i], hog);
                refineROI(roiCamShift, detected, roiHogDetected);
            }

            backProj = computeProbImage(sequence[i], roiCamShift, hist, detected);


            ///-------Test-Camshift--------------------///

            rectCamShift.resize(roiCamShift.size());

            for(size_t j=0;j<roiCamShift.size();j++)
            {
                /*
                std::cout<<roiCamShift[j]<<std::endl;
                cv::rectangle(backProj[j], roiCamShift[j], cv::Scalar( 255, 0, 0), 2, 8, 0 ); //DEBUG
                cv::imshow("before camshift", backProj[j]);
                cv::waitKey(0);
                */

                rectCamShift[j] = cv::CamShift(backProj[j], roiCamShift[j], cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
                rectCamShift[j].points(rect_points);

                for(int k = 0; k < 4; k++)
                    cv::line(sequence[i], rect_points[k], rect_points[(k+1)%4], cv::Scalar( 0, 0, 255), 2, 8);
            }
            ///----------------------------------------///

            //-----------------Representation----------------------------------//

            //dessin du rectangle


            for(size_t j=0;j<roiCamShift.size();j++)
                cv::rectangle(sequence[i], roiCamShift[j], cv::Scalar( 255, 0, 0), 2, 8, 0 );

            //affichage de la video
            cv::imshow("Video", sequence[i]);
        }



        //------------------CLEAR-VARIABLES------------------------------------//

        detectedPedestrian.clear();
        featuresDetected.clear();
        boxes.clear();

        previousSequenceGray.release();

        flow.release();
        imGray.release();
        imGrayPrev.release();

        roiHogDetected.clear();
        backProj.clear();

        //------------------CONDITIONS-ARRET-----------------------------------//

        if (cv::waitKey(66) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            return 0;
        }
    }

    cv::waitKey(0);
    return 0;
}
