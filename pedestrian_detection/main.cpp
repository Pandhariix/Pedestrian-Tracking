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

enum choiceAlgo{HOG_TEMPLATE_TRACKING,
                HOG_GOODFEATURESTOTRACK_LK,
                OPT_FLOW_FARNEBACK};

enum formatVideo{SEQUENCE_IMAGE,
                 VIDEO,
                 NOT_DEFINED};


/// Detection de l'algorithme utilisé

choiceAlgo detectAlgo(std::string argument)
{
    if(argument.compare("template_tracking") == 0)
        return HOG_TEMPLATE_TRACKING;

    else if(argument.compare("LK_tracking") == 0)
        return HOG_GOODFEATURESTOTRACK_LK;

    else if(argument.compare("farneback") == 0)
        return OPT_FLOW_FARNEBACK;

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




/// Detection via le HOG detector de opencv des piétons

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




/// Main

int main(int argc, char *argv[])
{

    //test pour savoir si l'utilisateur a renseigne un parametre
    if(argc <= 3)
    {
        std::cout<<"---------------------------------------"<<std::endl<<
                   "Veuillez rentrer la methode choisie :  "<<std::endl<<
                   "- template_tracking"                    <<std::endl<<
                   "- LK_tracking"                          <<std::endl<<
                   "- farneback"                <<std::endl<<std::endl<<
                   "Le type de format : "                   <<std::endl<<
                   "- video"                                <<std::endl<<
                   "- image"                     <<std::endl<<std::endl<<
                   "Le nom du fichier d'input"              <<std::endl<<
                   "---------------------------------------"<<std::endl;
        std::exit(EXIT_FAILURE);
    }


    //variables images et masque
    choiceAlgo algo;
    formatVideo format;
    std::string inputFileName(argv[3]);
    int nbTrames = 501;

    std::vector<cv::Mat> sequence;     //the sequence of images for the video
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

    //acquisition de la video
    algo = detectAlgo(std::string(argv[1]));
    format = detectFormat(std::string(argv[2]));

    if(format == SEQUENCE_IMAGE)
        sequence.resize(nbTrames);
    else if(format == VIDEO)
        std::cout<<"video";

    sequence = extractVideoData(format, inputFileName, nbTrames);
    sequenceGray.resize(sequence.size());

    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);




    //traitement sur la video
    for(int i=0;i<nbTrames;i++)
    {
        cv::cvtColor(sequence[i], sequenceGray[i], CV_BGR2GRAY);

        if(i>0)
            previousSequenceGray = sequenceGray[i-1];
        else
            previousSequenceGray = sequenceGray[i];




        /// HOG + Good Features to track + LK

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




        /// HOG + Template tracking

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





        /// HOG et optical flow farneback

        else if(algo == OPT_FLOW_FARNEBACK)
        {
            if(i!=0)
            {
                flow = cv::Mat::zeros(sequence[i].size(), CV_32FC2);
                cv::cvtColor(sequence[i], imGray, CV_BGR2GRAY);
                cv::cvtColor(sequence[i-1], imGrayPrev, CV_BGR2GRAY);

                cv::calcOpticalFlowFarneback(imGrayPrev, imGray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);


                //--------Representation--------------------

                drawOptFlowMap(flow, imGrayPrev, 16, CV_RGB(0, 255, 0)); //dessin test

                //affichage de la video
                cv::imshow("Video", imGrayPrev);
            }
        }



        //clear des variables
        detectedPedestrian.clear();
        featuresDetected.clear();
        boxes.clear();

        previousSequenceGray.release();

        flow.release();
        imGray.release();
        imGrayPrev.release();

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
