#pragma once

#include "framework.h"

enum class RegistrationMethod
{
    Phase,
    Orb,
    Akaze,
    Sift,
    Ecc
};

struct RegistrationResult
{
    bool success = false;
    CString methodName;
    CString message;
    int keypointsSrc = 0;
    int keypointsTarget = 0;
    int matches = 0;
    int inliers = 0;
    double rmse = 0.0;
    double elapsedMs = 0.0;
    double workingScale = 1.0;
    double angleDegrees = 0.0;
    cv::Point2d translation = cv::Point2d(0.0, 0.0);
    cv::Matx33d rigidTransform = cv::Matx33d::eye();
    cv::Mat stitchedImage;
};

class CRegistrationEngine
{
public:
    static CString GetMethodName(RegistrationMethod method);
    static RegistrationResult RegisterAndStitch(
        const cv::Mat& srcImage,
        const cv::Mat& targetImage,
        RegistrationMethod method);
};
