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
    if(argc <= 3)
    {
        std::cout<<"----------------------------------------------"<<std::endl<<
                   "---PedestrianTracker---"            <<std::endl<<std::endl<<
                   "Veuillez rentrer le type de format : "         <<std::endl<<
                   "- video"                                       <<std::endl<<
                   "- image"                            <<std::endl<<std::endl<<
                   "Le chemin/nom du fichier d'input"              <<std::endl<<
                   "Le type de tracking utilise :"                 <<std::endl<<
                   "- blob_tracking"                               <<std::endl<<
                   "- camshift_tracking"                <<std::endl<<std::endl<<
                   "-----------------------------------------------"<<std::endl;
        std::exit(EXIT_FAILURE);
    }

    //------------------VARIABLES--------------------//
    ArgParser parser(argc, std::string(argv[1]), std::string(argv[2]), std::string(argv[3]));

    std::vector<cv::Mat> sequence;
    cv::Mat previousSequence;
    int nbTrames = 501;
    double fps = 15;
    int trackingLimit = 0;
    trackingAlgo algorithm;

    //detection de l'algorithme de tracking
    algorithm = parser.selectedAlgorithm();

    //extraction de la video
    parser.extractVideo(sequence, nbTrames, fps);

    BackgroundSubSegmenter backSub;
    PedestrianBuilder builder(sequence[0].cols, sequence[0].rows);
    Tracker tracker(sequence[0].cols, sequence[0].rows);

    std::vector<cv::Rect> pedestrianSubDetected;
    std::vector<Pedestrian> pedestrian;
    std::vector<BlobPedestrian> blobPedestrian;



    //------------------TRAITEMENT-------------------//


    switch (algorithm) {

    case BLOB_TRACKING:
    {
        std::cout<<"PEDESTRIAN TRACKER"<<std::endl;
        std::cout<<"------------------"<<std::endl;
        std::cout<<"  Blob tracking"   <<std::endl;
        std::cout<<"------------------"<<std::endl<<std::endl;


        for(int i=0;i<nbTrames;i++)
        {
            if(i%10 == 0)
            {
                backSub.detectPedestrians(sequence[i], pedestrianSubDetected);
                tracker.createFeatures(sequence[i], pedestrianSubDetected, blobPedestrian);
            }
            else
            {
                tracker.trackFeatures(previousSequence, sequence[i], blobPedestrian);
            }

            previousSequence = sequence[i];

            for(size_t j=0;j<blobPedestrian.size();j++)
            {
                cv::rectangle(sequence[i], blobPedestrian[j].window, cv::Scalar( 0, 0, 255), 2, 8, 0 );

                /*
                for(size_t k=0;j<blobPedestrian[j].features.size();j++)
                {
                    cv::circle(sequence[i], blobPedestrian[j].features[k], 1, cv::Scalar(0,255,255), 1, 24);
                }
                */
            }

            cv::imshow("PedestrianTracker", sequence[i]);



            //--------------CONDITIONS-ARRET-------------//

            if (cv::waitKey((int)(1000/fps)) == 27) //jouer la video au bon framerate, quitter le programme si echap est pressee
            {
                std::cout << "esc enfoncee, fin du programme" << std::endl;
                return 0;
            }
        }


        break;
    }

    case CAMSHIFT_TRACKING:
    {

        std::cout<<"PEDESTRIAN TRACKER"<<std::endl;
        std::cout<<"------------------"<<std::endl;
        std::cout<<"Camshift tracking" <<std::endl;
        std::cout<<"------------------"<<std::endl<<std::endl;
        std::cout<<"Veuillez rentrer la limite de pietons trackes : ";
        std::cin>> trackingLimit;
        std::cout<<std::endl<<std::endl;

        for(int i=0;i<nbTrames;i++)
        {
            if(i%10 == 0)
            {
                backSub.detectPedestrians(sequence[i], pedestrianSubDetected);
                builder.detectNewPedestrian(pedestrian, pedestrianSubDetected);
            }

            builder.buildPedestrian(pedestrian, sequence[i]);
            tracker.camshift(pedestrian);


            for(size_t j=0;j<pedestrian.size() && j < static_cast<size_t>(trackingLimit);j++)
            {
                cv::rectangle(sequence[i], pedestrian[j].window, cv::Scalar( 0, 0, 255), 2, 8, 0 );
                cv::putText(sequence[i], "P"+std::to_string(j), cv::Point(pedestrian[j].window.x, pedestrian[j].window.y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
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


        break;
    }

    default:
        std::cout<<"Erreur, cet algorithme n'existe pas"<<std::endl;
        std::exit(EXIT_FAILURE);
        break;
    }

    cv::waitKey(0);
    return 0;
}

