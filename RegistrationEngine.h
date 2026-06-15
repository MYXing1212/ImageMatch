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

struct MatchLineSegment
{
    cv::Point2d start = cv::Point2d(0.0, 0.0);
    cv::Point2d end = cv::Point2d(0.0, 0.0);
};

struct RegistrationResult
{
    bool success = false;
    bool usedAcceleration = false;
    bool hasFeatureVisualization = false;
    CString methodName;
    CString message;
    CString accelerationProfile;
    CString logText;
    int keypointsSrc = 0;
    int keypointsTarget = 0;
    int matches = 0;
    int inliers = 0;
    int workingWidth = 0;
    int workingHeight = 0;
    double rmse = 0.0;
    double elapsedMs = 0.0;
    double workingScale = 1.0;
    double angleDegrees = 0.0;
    cv::Point2d translation = cv::Point2d(0.0, 0.0);
    cv::Matx33d rigidTransform = cv::Matx33d::eye();
    cv::Mat stitchedImage;
    std::vector<cv::Point2d> srcVisualizationPoints;
    std::vector<cv::Point2d> targetVisualizationPoints;
    std::vector<MatchLineSegment> srcVisualizationLines;
    std::vector<MatchLineSegment> targetVisualizationLines;
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
