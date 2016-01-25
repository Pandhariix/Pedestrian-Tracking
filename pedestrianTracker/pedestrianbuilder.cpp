#include "pedestrianbuilder.h"

PedestrianBuilder::PedestrianBuilder(int videoWidth, int videoHeight)
{
    this->videoWidth = videoWidth;
    this->videoHeight = videoHeight;
}


void PedestrianBuilder::detectNewPedestrian(std::vector<Pedestrian> &pedestrian, std::vector<cv::Rect> pedestrianSubDetected)
{
    if(pedestrianSubDetected.size() == 0)
        return;

    cv::Rect intersect;
    Pedestrian newPedestrian;

    newPedestrian.known = false;
    newPedestrian.histogram.release();
    newPedestrian.histogramImage.release();
    newPedestrian.backProj.release();

    for(size_t i=0;i<pedestrianSubDetected.size();i++)
    {
        for(size_t j=0;j<pedestrian.size();j++)
        {
            intersect = pedestrianSubDetected[i] & pedestrian[j].window;

            if(intersect.area() != 0)
                break;
        }

        if(intersect.area() == 0 && pedestrian.size() < pedestrianSubDetected.size() && (pedestrianSubDetected[i].width/pedestrianSubDetected[i].height) <= 0.33)
        {
            newPedestrian.window = pedestrianSubDetected[i];

            //TODO meilleur centrage
            newPedestrian.window.x += newPedestrian.window.width/3;
            newPedestrian.window.width = newPedestrian.window.width/4;
            newPedestrian.window.y += newPedestrian.window.height/3;
            newPedestrian.window.height = newPedestrian.window.height/4;
            //
            pedestrian.push_back(newPedestrian);
        }
    }
}



void PedestrianBuilder::buildPedestrian(std::vector<Pedestrian> &pedestrian, cv::Mat sequence)
{
    int smin = 30;
    int vmin = 10;
    int vmax = 256;
    cv::Mat mask;
    cv::Mat hsv;
    cv::Mat hue;
    cv::Mat histimg = cv::Mat::zeros(200, 320, CV_8UC3);
    int channels[] = {0,0};
    int hbins = 30;                          // On a 30 niveaux de hue
    int histSize = MAX( hbins, 2 );
    float hue_range[] = { 0, 180 };          // le hue varie de 0 a 179 (cf cv::cvtColor)
    const float* range = { hue_range };

    cv::cvtColor(sequence, hsv, CV_BGR2HSV);
    hue.create(hsv.size(), hsv.depth());
    cv::mixChannels(&hsv, 1, &hue, 1, channels, 1);
    cv::inRange(hsv, cv::Scalar(0, smin, MIN(vmin,vmax)), cv::Scalar(180, 256, MAX(vmin, vmax)), mask);


    for(size_t i=0;i<pedestrian.size();i++)
    {
        if(!pedestrian[i].known)
        {
            cv::Mat roi(hue, pedestrian[i].window);
            cv::Mat maskroi(mask, pedestrian[i].window);

            cv::calcHist(&roi, 1, 0, maskroi, pedestrian[i].histogram, 1, &histSize, &range, true, false);
            cv::normalize(pedestrian[i].histogram, pedestrian[i].histogram, 0, 255, cv::NORM_MINMAX);

            histimg = cv::Scalar::all(0);
            int binW = histimg.cols / histSize;
            cv::Mat buf(1, histSize, CV_8UC3);

            for( int j = 0; j < histSize; j++ )
                buf.at<cv::Vec3b>(j) = cv::Vec3b(cv::saturate_cast<uchar>(j*180./histSize), 255, 255);

            cv::cvtColor(buf, buf, cv::COLOR_HSV2BGR);

            for( int j = 0; j < histSize; j++ )
            {
                int val = cv::saturate_cast<int>(pedestrian[i].histogram.at<float>(j)*histimg.rows/255);
                cv::rectangle( histimg, cv::Point(j*binW,histimg.rows),
                           cv::Point((j+1)*binW,histimg.rows - val),
                           cv::Scalar(buf.at<cv::Vec3b>(j)), -1, 8 );
            }

            pedestrian[i].histogramImage = histimg;
            pedestrian[i].known = true;

            roi.release();
            maskroi.release();
                buf.release();
        }

        cv::calcBackProject(&hue, 1, 0, pedestrian[i].histogram, pedestrian[i].backProj, &range);
        pedestrian[i].backProj &= mask;
    }
}



