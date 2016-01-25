#ifndef BLOBPEDESTRIANSTRUCTURE
#define BLOBPEDESTRIANSTRUCTURE

#include <opencv2/opencv.hpp>

typedef struct BlobPedestrian
{
    std::vector<cv::Point2f> features;
    cv::Rect window;

}BlobPedestrian;

#endif // BLOBPEDESTRIANSTRUCTURE

