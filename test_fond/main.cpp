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
    /*
    if(argc != 3)
    {
        std::cout<<"unable to launch the program, please enter correct input arguments"<<std::endl;
        return -1;
    }

    //init the matrix
    cv::Mat imageFond;
    cv::Mat imageCompare;
    cv::Mat imageFondGray;
    cv::Mat imageCompareGray;
    cv::Mat imageFondBinary;
    cv::Mat imageCompareBinary;
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

    //convert both images to binary

    cv::threshold( imageFondGray, imageFondBinary, 20, 255,1);
    cv::threshold(imageCompareGray, imageCompareBinary, 20, 255, 1);

    //difference between the two images
    cv::absdiff(imageFondBinary, imageCompareBinary, imageDiff);

    //display
    cv::namedWindow( "test_fond", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "test_fond", imageDiff);

    cv::waitKey(0);
    return(0);
    */

    ///TEST TRACKING OF A RED OBJECT PROGRAM
    /*
    cv::VideoCapture cap(0); //capture the video from webcam

    if ( !cap.isOpened() )  // if not success, exit program
    {
         std::cout << "Cannot open the web cam" << std::endl;
         return -1;
    }

    cv::namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    int iLowH = 170;
    int iHighH = 179;

    int iLowS = 150;
    int iHighS = 255;

    int iLowV = 60;
    int iHighV = 255;

    //Create trackbars in "Control" window
    cv::createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    cv::createTrackbar("HighH", "Control", &iHighH, 179);

    cv::createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    cv::createTrackbar("HighS", "Control", &iHighS, 255);

    cv::createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
    cv::createTrackbar("HighV", "Control", &iHighV, 255);

    int iLastX = -1;
    int iLastY = -1;

    //Capture a temporary image from the camera
    cv::Mat imgTmp;
    cap.read(imgTmp);

    //Create a black image with the size as the camera output
    cv::Mat imgLines = cv::Mat::zeros( imgTmp.size(), CV_8UC3 );;


    while (true)
    {
        cv::Mat imgOriginal;

        bool bSuccess = cap.read(imgOriginal); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
            std::cout << "Cannot read a frame from video stream" << std::endl;
            break;
        }

        cv::Mat imgHSV;

        cv::cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

        cv::Mat imgThresholded;

        cv::inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

        //morphological opening (removes small objects from the foreground)
        cv::erode(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );
        cv::dilate( imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );

        //morphological closing (removes small holes from the foreground)
        cv::dilate( imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );
        cv::erode(imgThresholded, imgThresholded, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)) );

        //Calculate the moments of the thresholded image
        cv::Moments oMoments = cv::moments(imgThresholded);

        double dM01 = oMoments.m01;
        double dM10 = oMoments.m10;
        double dArea = oMoments.m00;

         // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero
        if (dArea > 10000)
        {
            //calculate the position of the ball
            int posX = dM10 / dArea;
            int posY = dM01 / dArea;

            if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
            {
                //Draw a red line from the previous point to the current point
                cv::line(imgLines, cv::Point(posX, posY), cv::Point(iLastX, iLastY), cv::Scalar(0,0,255), 2);
            }

            iLastX = posX;
            iLastY = posY;
        }

        cv::imshow("Thresholded Image", imgThresholded); //show the thresholded image

        imgOriginal = imgOriginal + imgLines;
        cv::imshow("Original", imgOriginal); //show the original image

        if (cv::waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }

    return 0;
    */

    ///PERSONAL TEST

    if(argc != 2)
    {
        std::cout<<"wrong number of input arguments"<<std::endl;
        return -1;
    }

    cv::VideoCapture cap(0); //capture the video from webcam

    if ( !cap.isOpened() )  // if not success, exit program
    {
         std::cout << "Cannot open the web cam" << std::endl;
         return -1;
    }

    int thres = (int)(argv[1]);

    cv::Mat imgOriginal;
    cv::Mat imgOriginalGray;
    cv::Mat img;
    cv::Mat imgGray;
    cv::Mat imgDiff;

    bool bSuccess = false;

    bSuccess = cap.read(imgOriginal);
    if (!bSuccess) //if not success, exit
    {
        std::cout << "Cannot read a frame from video stream" << std::endl;
        return 0;
    }

    cv::cvtColor(imgOriginal, imgOriginalGray, CV_BGR2GRAY); //Convert the captured frame from BGR to gray


    while(true)
    {


        bSuccess = cap.read(img); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
            std::cout << "Cannot read a frame from video stream" << std::endl;
            break;
        }

        cv::cvtColor(img, imgGray, CV_BGR2GRAY); //Convert the captured frame from BGR to gray

        cv::absdiff(imgGray, imgOriginalGray, imgDiff);

        cv::threshold(imgDiff, imgDiff, thres, 255, 4);

        cv::imshow("Image", imgDiff);

        if (cv::waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }

    return 0;
}
