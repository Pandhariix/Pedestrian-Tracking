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

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <iostream>

#include <argparser.h>
#include <backgroundsubsegmenter.h>
#include <pedestrianbuilder.h>
#include <tracker.h>


//-------------------------------------------------------------------------//
//******************************MAIN***************************************//
//-------------------------------------------------------------------------//

int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        std::cout<<"----------------------------------------------"<<std::endl<<
                   "---PedestrianTracker---"            <<std::endl<<std::endl<<
                   "Veuillez rentrer le type de format : "         <<std::endl<<
                   "- video"                                       <<std::endl<<
                   "- image"                            <<std::endl<<std::endl<<
                   "Le chemin/nom du fichier d'input"              <<std::endl<<
                   "-----------------------------------------------"<<std::endl;
        std::exit(EXIT_FAILURE);
    }

    //------------------VARIABLES--------------------//
    ArgParser parser(argc, std::string(argv[1]), std::string(argv[2]));
    BackgroundSubSegmenter backSub;
    PedestrianBuilder builder;
    Tracker tracker;

    std::vector<cv::Mat> sequence;
    int nbTrames = 501;
    double fps = 15;

    std::vector<cv::Rect> pedestrianSubDetected;
    std::vector<Pedestrian> pedestrian;

    //------------------VIDEO------------------------//
    parser.extractVideo(sequence, nbTrames, fps);




    //------------------TRAITEMENT-------------------//

    for(int i=0;i<nbTrames;i++)
    {
        if(i%20 == 0)
        {
            backSub.detectPedestrians(sequence[i], pedestrianSubDetected);
            builder.detectNewPedestrian(pedestrian, pedestrianSubDetected);
        }

        builder.buildPedestrian(pedestrian, sequence[i]);
        tracker.camshift(pedestrian);

        for(size_t j=0;j<pedestrian.size() && j<3;j++)
        {
            cv::rectangle(sequence[i], pedestrian[j].window, cv::Scalar( 0, 0, 255), 2, 8, 0 );
            cv::putText(sequence[i], "P"+std::to_string(j), cv::Point(pedestrian[j].window.x, pedestrian[j].window.y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

            //cv::imshow("histogram"+std::to_string(j), pedestrian[j].histogramImage);
            cv::imshow("backproj "+std::to_string(j), pedestrian[j].backProj);
        }

        cv::imshow("PedestrianTracker", sequence[i]);


        pedestrianSubDetected.clear();

        //--------------CONDITIONS-ARRET-------------//

        if (cv::waitKey((int)(1000/fps)) == 27) //jouer la video au bon framerate, quitter le programme si echap est pressee
        {
            std::cout << "esc enfoncee, fin du programme" << std::endl;
            return 0;
        }
    }

    cv::waitKey(0);
    return 0;
}

