#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;

int main( int argc, char** argv )
{
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
}
