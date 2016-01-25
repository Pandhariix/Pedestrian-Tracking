#include "argparser.h"

ArgParser::ArgParser()
{
    std::cout<<"----------------------------------------------"<<std::endl<<
               "---PedestrianTracker---"            <<std::endl<<std::endl<<
               "Veuillez rentrer le type de format : "         <<std::endl<<
               "- video"                                       <<std::endl<<
               "- image"                            <<std::endl<<std::endl<<
               "Le chemin/nom du fichier d'input"              <<std::endl<<
               "Le type de tracking utilisé :"                 <<std::endl<<
               "- blob_tracking"                               <<std::endl<<
               "- camshift_tracking"                <<std::endl<<std::endl<<
               "-----------------------------------------------"<<std::endl;
    std::exit(EXIT_FAILURE);
}

ArgParser::ArgParser(const int argc, std::string format, std::string file, std::string tracking_algorithm)
{
    if(argc != 4)
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

    this->args.resize(3);
    this->args[0] = format;
    this->args[1] = file;
    this->args[2] = tracking_algorithm;

    this->detectFormat();
    this->detectTrackingAlgorithm();
}



void ArgParser::detectFormat()
{
    if(this->args[0].compare("image") == 0)
        this->formatType = SEQUENCE_IMAGE;

    else if(this->args[0].compare("video") == 0)
        this->formatType = SEQUENCE_VIDEO;

    else
        this->formatType = UNDEFINED_FORMAT;
}


void ArgParser::detectTrackingAlgorithm()
{
    if(this->args[2].compare("blob_tracking") == 0)
        this->algorithm = BLOB_TRACKING;

    else if(this->args[2].compare("camshift_tracking") == 0)
        this->algorithm = CAMSHIFT_TRACKING;

    else
        this->algorithm = UNDEFINED_ALGORITHM;
}


void ArgParser::extractVideo(std::vector<cv::Mat> &sequence, int &nbTrames, double &fps)
{
    sequence.clear();

    switch (this->formatType){


    case SEQUENCE_IMAGE:
    {
        sequence.resize(nbTrames);

        for(int i=0;i<nbTrames;i++)
        {
            std::stringstream nameTrame;
            if(i<10)
            {
                nameTrame << this->args[1] << "_000" << i << ".jpeg";
            }
            else if(i<100)
            {
                nameTrame << this->args[1] << "_00" << i << ".jpeg";
            }
            else
            {
                nameTrame << this->args[1] << "_0" << i << ".jpeg";
            }

            std::cout<<nameTrame.str()<<std::endl;

            sequence[i] = cv::imread(nameTrame.str());
        }
        break;
    }

    case SEQUENCE_VIDEO:
    {
        cv::VideoCapture capture;
        cv::Mat frame;

        capture.open(this->args[1]);

        if(capture.isOpened())
        {
            fps = capture.get(CV_CAP_PROP_FPS);

            while(true)
            {
                if(!capture.read(frame))
                    break;

                sequence.push_back(frame);
                frame.release();
            }

            capture.release();
            nbTrames = sequence.size();
        }
        else
        {
            std::cout<<"Impossible de lire la video : "<<this->args[1]<<std::endl;
        }
        break;
    }

    case UNDEFINED_FORMAT:
        std::cout<<"Argument incorrect, veuillez entrer 'image' ou 'video'"<<std::endl;
        std::exit(EXIT_FAILURE);
        break;

    default:
        std::cout<<"Argument incorrect, veuillez entrer 'image' ou 'video'"<<std::endl;
        std::exit(EXIT_FAILURE);
        break;

    }
}



trackingAlgo ArgParser::selectedAlgorithm()
{
    return this->algorithm;
}
