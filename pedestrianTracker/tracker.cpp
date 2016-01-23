#include "tracker.h"

Tracker::Tracker()
{

}


void Tracker::camshift(std::vector<Pedestrian> &pedestrian)
{
    for(size_t i=0;i<pedestrian.size();i++)
    {
        cv::CamShift(pedestrian[i].backProj, pedestrian[i].window, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
    }
}
