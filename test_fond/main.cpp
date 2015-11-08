#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;

int main( int argc, char** argv )
{

    /// INITIAL TEST PROGRAM
    /*
    if( argc != 2)
    {
     std::cout <<" Usage: display_image ImageToLoadAndDisplay" << std::endl;
     return -1;
    }

    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::Mat imageGray;

    //convert RGB to gray
    cv::cvtColor(image, imageGray, CV_BGR2GRAY);


    cv::Mat binary;

    //convert gray to binary
    cv::threshold( imageGray, binary, 20, 255,1);


    cv::Mat imageFinale;

    // Create a structuring element
    int erosion_size = 8;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size) );


    cv::erode(binary,imageFinale,element);
    cv::dilate(imageFinale,imageFinale,element);

    cv::namedWindow( "Original", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Original", image);                   // Show our image inside it.

    cv::namedWindow( "Finale", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Finale", imageFinale);

    cv::waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
    */

    ///OFFICIAL TEST FOND PROGRAM
    if(argc != 3)
    {
        std::cout<<"unable to launch the program, please enter correct input arguments"<<std::endl;
    }

    //init the matrix
    cv::Mat imageFond;
    cv::Mat imageCompare;
    cv::Mat imageFondGray;
    cv::Mat imageCompareGray;
    cv::Mat imageDiff;

    imageFond = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    imageCompare = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    if(!imageFond.data || !imageCompare.data)
    {
        std::cout<<"at least one of the images can not be read"<<std::endl;
        return -1;
    }

    //convert both images to grayscale

    cv::cvtColor(imageFond, imageFondGray, CV_BGR2GRAY);
    cv::cvtColor(imageCompare, imageCompareGray, CV_BGR2GRAY);

    //difference between the two images
    cv::absdiff(imageFondGray, imageCompareGray, imageDiff);

    //display
    cv::namedWindow( "test_fond", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "test_fond", imageDiff);

    cv::waitKey(0);
    return(0);


}
