#include "framework.h"
#include "RegistrationEngine.h"

namespace
{
struct WorkingPair
{
    cv::Mat src;
    cv::Mat target;
    double scale = 1.0;
};

cv::Mat ConvertToGray8(const cv::Mat& image)
{
    if (image.empty())
    {
        return {};
    }

    cv::Mat gray;
    if (image.channels() == 1)
    {
        gray = image.clone();
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    if (gray.depth() != CV_8U)
    {
        cv::Mat normalized;
        gray.convertTo(normalized, CV_8U);
        return normalized;
    }
    return gray;
}

WorkingPair PrepareWorkingPair(const cv::Mat& srcImage, const cv::Mat& targetImage, int maxDimension)
{
    WorkingPair pair;
    const int maxInputDimension = std::max(
        std::max(srcImage.cols, srcImage.rows),
        std::max(targetImage.cols, targetImage.rows));
    pair.scale = (maxInputDimension > maxDimension)
        ? static_cast<double>(maxDimension) / static_cast<double>(maxInputDimension)
        : 1.0;

    if (pair.scale < 1.0)
    {
        cv::resize(srcImage, pair.src, cv::Size(), pair.scale, pair.scale, cv::INTER_AREA);
        cv::resize(targetImage, pair.target, cv::Size(), pair.scale, pair.scale, cv::INTER_AREA);
    }
    else
    {
        pair.src = srcImage.clone();
        pair.target = targetImage.clone();
    }

    return pair;
}

void PadToSameSize(const cv::Mat& src, const cv::Mat& target, cv::Mat& srcOut, cv::Mat& targetOut)
{
    const int rows = std::max(src.rows, target.rows);
    const int cols = std::max(src.cols, target.cols);
    cv::copyMakeBorder(src, srcOut, 0, rows - src.rows, 0, cols - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::copyMakeBorder(target, targetOut, 0, rows - target.rows, 0, cols - target.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
}

cv::Matx33d BuildTranslationTransform(double tx, double ty)
{
    return cv::Matx33d(
        1.0, 0.0, tx,
        0.0, 1.0, ty,
        0.0, 0.0, 1.0);
}

cv::Matx33d BuildRigidTransform(const cv::Mat& affine)
{
    cv::Mat affine64;
    affine.convertTo(affine64, CV_64F);

    const double a00 = affine64.at<double>(0, 0);
    const double a01 = affine64.at<double>(0, 1);
    const double a10 = affine64.at<double>(1, 0);
    const double a11 = affine64.at<double>(1, 1);
    const double tx = affine64.at<double>(0, 2);
    const double ty = affine64.at<double>(1, 2);

    const double scaleX = std::sqrt(a00 * a00 + a10 * a10);
    const double scaleY = std::sqrt(a01 * a01 + a11 * a11);
    const double scale = std::max(scaleX, scaleY);
    if (scale < 1e-8)
    {
        throw std::runtime_error("Rigid transform scale is too small.");
    }

    const double cosTheta = a00 / scale;
    const double sinTheta = a10 / scale;

    return cv::Matx33d(
        cosTheta, -sinTheta, tx,
        sinTheta, cosTheta, ty,
        0.0, 0.0, 1.0);
}

cv::Matx33d ScaleTransformToOriginal(const cv::Matx33d& workingTransform, double scale)
{
    if (std::abs(scale - 1.0) < 1e-8)
    {
        return workingTransform;
    }

    const cv::Matx33d scaleMatrix(
        scale, 0.0, 0.0,
        0.0, scale, 0.0,
        0.0, 0.0, 1.0);
    const cv::Matx33d inverseScaleMatrix(
        1.0 / scale, 0.0, 0.0,
        0.0, 1.0 / scale, 0.0,
        0.0, 0.0, 1.0);
    return inverseScaleMatrix * workingTransform * scaleMatrix;
}

void UpdatePoseFields(RegistrationResult& result)
{
    result.translation = cv::Point2d(result.rigidTransform(0, 2), result.rigidTransform(1, 2));
    result.angleDegrees = std::atan2(result.rigidTransform(1, 0), result.rigidTransform(0, 0)) * 180.0 / CV_PI;
}

std::vector<cv::Point2d> TransformCorners(const cv::Size& size, const cv::Matx33d& transform)
{
    const std::vector<cv::Point2d> corners = {
        {0.0, 0.0},
        {static_cast<double>(size.width), 0.0},
        {static_cast<double>(size.width), static_cast<double>(size.height)},
        {0.0, static_cast<double>(size.height)}};

    std::vector<cv::Point2d> transformedCorners;
    transformedCorners.reserve(corners.size());
    for (const auto& corner : corners)
    {
        const cv::Vec3d point(corner.x, corner.y, 1.0);
        const cv::Vec3d mapped = transform * point;
        transformedCorners.emplace_back(mapped[0], mapped[1]);
    }
    return transformedCorners;
}

cv::Mat StitchImages(const cv::Mat& srcImage, const cv::Mat& targetImage, const cv::Matx33d& srcToTarget)
{
    const std::vector<cv::Point2d> srcCorners = TransformCorners(srcImage.size(), srcToTarget);
    const std::vector<cv::Point2d> targetCorners = {
        {0.0, 0.0},
        {static_cast<double>(targetImage.cols), 0.0},
        {static_cast<double>(targetImage.cols), static_cast<double>(targetImage.rows)},
        {0.0, static_cast<double>(targetImage.rows)}};

    double minX = 0.0;
    double minY = 0.0;
    double maxX = static_cast<double>(targetImage.cols);
    double maxY = static_cast<double>(targetImage.rows);

    for (const auto& pt : srcCorners)
    {
        minX = std::min(minX, pt.x);
        minY = std::min(minY, pt.y);
        maxX = std::max(maxX, pt.x);
        maxY = std::max(maxY, pt.y);
    }

    for (const auto& pt : targetCorners)
    {
        minX = std::min(minX, pt.x);
        minY = std::min(minY, pt.y);
        maxX = std::max(maxX, pt.x);
        maxY = std::max(maxY, pt.y);
    }

    const int offsetX = static_cast<int>(std::floor(-minX));
    const int offsetY = static_cast<int>(std::floor(-minY));
    const int canvasWidth = static_cast<int>(std::ceil(maxX + offsetX));
    const int canvasHeight = static_cast<int>(std::ceil(maxY + offsetY));

    cv::Mat canvas(canvasHeight, canvasWidth, CV_8UC3, cv::Scalar::all(0));
    cv::Mat maskSrc(srcImage.rows, srcImage.cols, CV_8UC1, cv::Scalar::all(255));
    cv::Mat warpedMask(canvasHeight, canvasWidth, CV_8UC1, cv::Scalar::all(0));

    const cv::Matx33d translation(
        1.0, 0.0, static_cast<double>(offsetX),
        0.0, 1.0, static_cast<double>(offsetY),
        0.0, 0.0, 1.0);
    const cv::Matx33d srcToCanvas = translation * srcToTarget;
    const cv::Mat affineTransform = (cv::Mat_<double>(2, 3)
        << srcToCanvas(0, 0), srcToCanvas(0, 1), srcToCanvas(0, 2),
           srcToCanvas(1, 0), srcToCanvas(1, 1), srcToCanvas(1, 2));

    cv::warpAffine(
        srcImage,
        canvas,
        affineTransform,
        canvas.size(),
        cv::INTER_LINEAR,
        cv::BORDER_TRANSPARENT);
    cv::warpAffine(
        maskSrc,
        warpedMask,
        affineTransform,
        canvas.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar::all(0));

    const cv::Rect targetRoi(offsetX, offsetY, targetImage.cols, targetImage.rows);
    cv::Mat canvasRoi = canvas(targetRoi);
    const cv::Mat overlapMask = warpedMask(targetRoi);

    cv::Mat nonOverlapMask;
    cv::bitwise_not(overlapMask, nonOverlapMask);
    targetImage.copyTo(canvasRoi, nonOverlapMask);

    cv::Mat blended;
    cv::addWeighted(canvasRoi, 0.5, targetImage, 0.5, 0.0, blended);
    blended.copyTo(canvasRoi, overlapMask);

    return canvas;
}

double ComputeImageRmse(const cv::Mat& srcImage, const cv::Mat& targetImage, const cv::Matx33d& srcToTarget)
{
    const cv::Mat srcGray = ConvertToGray8(srcImage);
    const cv::Mat targetGray = ConvertToGray8(targetImage);
    if (srcGray.empty() || targetGray.empty())
    {
        return 0.0;
    }

    cv::Mat warpedSrc;
    cv::Mat warpedMask;
    cv::Mat srcMask(srcGray.rows, srcGray.cols, CV_8UC1, cv::Scalar::all(255));
    const cv::Mat affineTransform = (cv::Mat_<double>(2, 3)
        << srcToTarget(0, 0), srcToTarget(0, 1), srcToTarget(0, 2),
           srcToTarget(1, 0), srcToTarget(1, 1), srcToTarget(1, 2));

    cv::warpAffine(
        srcGray,
        warpedSrc,
        affineTransform,
        targetGray.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar::all(0));
    cv::warpAffine(
        srcMask,
        warpedMask,
        affineTransform,
        targetGray.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar::all(0));

    const int overlapPixels = cv::countNonZero(warpedMask);
    if (overlapPixels <= 0)
    {
        return 0.0;
    }

    cv::Mat diff;
    cv::absdiff(warpedSrc, targetGray, diff);
    cv::Mat diffFloat;
    diff.convertTo(diffFloat, CV_32F);
    cv::Mat squared;
    cv::multiply(diffFloat, diffFloat, squared);
    const cv::Scalar meanSquared = cv::mean(squared, warpedMask);
    return std::sqrt(meanSquared[0]);
}

RegistrationResult RunFeatureMethod(
    const cv::Mat& srcImage,
    const cv::Mat& targetImage,
    const CString& methodName,
    const cv::Ptr<cv::Feature2D>& detector,
    int matcherNorm,
    int maxDimension,
    float ratioThreshold,
    size_t maxMatchesToKeep)
{
    RegistrationResult result;
    result.methodName = methodName;

    const WorkingPair pair = PrepareWorkingPair(srcImage, targetImage, maxDimension);
    result.workingScale = pair.scale;

    const cv::Mat srcGray = ConvertToGray8(pair.src);
    const cv::Mat targetGray = ConvertToGray8(pair.target);

    std::vector<cv::KeyPoint> srcKeypoints;
    std::vector<cv::KeyPoint> targetKeypoints;
    cv::Mat srcDescriptors;
    cv::Mat targetDescriptors;
    detector->detectAndCompute(srcGray, cv::noArray(), srcKeypoints, srcDescriptors);
    detector->detectAndCompute(targetGray, cv::noArray(), targetKeypoints, targetDescriptors);

    result.keypointsSrc = static_cast<int>(srcKeypoints.size());
    result.keypointsTarget = static_cast<int>(targetKeypoints.size());
    if (srcDescriptors.empty() || targetDescriptors.empty())
    {
        result.message = L"Not enough features detected.";
        return result;
    }

    cv::BFMatcher matcher(matcherNorm);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(srcDescriptors, targetDescriptors, knnMatches, 2);

    std::vector<cv::DMatch> goodMatches;
    goodMatches.reserve(knnMatches.size());
    for (const auto& candidates : knnMatches)
    {
        if (candidates.size() < 2)
        {
            continue;
        }

        if (candidates[0].distance < ratioThreshold * candidates[1].distance)
        {
            goodMatches.push_back(candidates[0]);
        }
    }

    std::sort(goodMatches.begin(), goodMatches.end(), [](const cv::DMatch& left, const cv::DMatch& right)
    {
        return left.distance < right.distance;
    });
    if (goodMatches.size() > maxMatchesToKeep)
    {
        goodMatches.resize(maxMatchesToKeep);
    }

    result.matches = static_cast<int>(goodMatches.size());
    if (goodMatches.size() < 4)
    {
        result.message = L"Too few good matches for rigid transform estimation.";
        return result;
    }

    std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> targetPoints;
    srcPoints.reserve(goodMatches.size());
    targetPoints.reserve(goodMatches.size());
    for (const auto& match : goodMatches)
    {
        srcPoints.push_back(srcKeypoints[match.queryIdx].pt);
        targetPoints.push_back(targetKeypoints[match.trainIdx].pt);
    }

    cv::Mat inlierMask;
    const cv::Mat affine = cv::estimateAffinePartial2D(
        srcPoints,
        targetPoints,
        inlierMask,
        cv::RANSAC,
        3.0,
        2000,
        0.99,
        15);

    if (affine.empty())
    {
        result.message = L"RANSAC failed to estimate a stable transform.";
        return result;
    }

    result.inliers = cv::countNonZero(inlierMask);
    result.rigidTransform = ScaleTransformToOriginal(BuildRigidTransform(affine), pair.scale);
    UpdatePoseFields(result);
    result.success = true;
    return result;
}

RegistrationResult RunPhaseMethod(const cv::Mat& srcImage, const cv::Mat& targetImage)
{
    RegistrationResult result;
    result.methodName = L"PHASE";

    const WorkingPair pair = PrepareWorkingPair(srcImage, targetImage, 1600);
    result.workingScale = pair.scale;

    cv::Mat srcGray;
    cv::Mat targetGray;
    PadToSameSize(ConvertToGray8(pair.src), ConvertToGray8(pair.target), srcGray, targetGray);

    cv::Mat srcFloat;
    cv::Mat targetFloat;
    srcGray.convertTo(srcFloat, CV_32F);
    targetGray.convertTo(targetFloat, CV_32F);

    cv::Mat window;
    cv::createHanningWindow(window, srcFloat.size(), CV_32F);
    const cv::Point2d shift = cv::phaseCorrelate(srcFloat, targetFloat, window);

    const cv::Matx33d forward = BuildTranslationTransform(shift.x, shift.y);
    const cv::Matx33d backward = BuildTranslationTransform(-shift.x, -shift.y);
    const double forwardRmse = ComputeImageRmse(pair.src, pair.target, forward);
    const double backwardRmse = ComputeImageRmse(pair.src, pair.target, backward);

    result.rigidTransform = ScaleTransformToOriginal(
        (forwardRmse <= backwardRmse) ? forward : backward,
        pair.scale);
    UpdatePoseFields(result);
    result.success = true;
    return result;
}

RegistrationResult RunEccMethod(const cv::Mat& srcImage, const cv::Mat& targetImage)
{
    RegistrationResult result;
    result.methodName = L"ECC";

    const WorkingPair pair = PrepareWorkingPair(srcImage, targetImage, 1800);
    result.workingScale = pair.scale;

    cv::Mat srcGray;
    cv::Mat targetGray;
    PadToSameSize(ConvertToGray8(pair.src), ConvertToGray8(pair.target), srcGray, targetGray);

    cv::Mat srcFloat;
    cv::Mat targetFloat;
    srcGray.convertTo(srcFloat, CV_32F, 1.0 / 255.0);
    targetGray.convertTo(targetFloat, CV_32F, 1.0 / 255.0);

    constexpr int maxLevel = 2;
    std::vector<cv::Mat> srcPyramid;
    std::vector<cv::Mat> targetPyramid;
    cv::buildPyramid(srcFloat, srcPyramid, maxLevel);
    cv::buildPyramid(targetFloat, targetPyramid, maxLevel);

    cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);

    {
        cv::Mat coarseWindow;
        cv::createHanningWindow(coarseWindow, srcPyramid.back().size(), CV_32F);
        const cv::Point2d initialShift = cv::phaseCorrelate(srcPyramid.back(), targetPyramid.back(), coarseWindow);
        warp.at<float>(0, 2) = static_cast<float>(initialShift.x);
        warp.at<float>(1, 2) = static_cast<float>(initialShift.y);
    }

    for (int level = maxLevel; level >= 0; --level)
    {
        if (level < maxLevel)
        {
            warp.at<float>(0, 2) *= 2.0f;
            warp.at<float>(1, 2) *= 2.0f;
        }

        const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-5);
        cv::findTransformECC(
            targetPyramid[level],
            srcPyramid[level],
            warp,
            cv::MOTION_EUCLIDEAN,
            criteria,
            cv::noArray(),
            5);
    }

    result.rigidTransform = ScaleTransformToOriginal(BuildRigidTransform(warp), pair.scale);
    UpdatePoseFields(result);
    result.success = true;
    return result;
}

void FinalizeResult(RegistrationResult& result, const cv::Mat& srcImage, const cv::Mat& targetImage)
{
    if (!result.success)
    {
        return;
    }

    result.rmse = ComputeImageRmse(srcImage, targetImage, result.rigidTransform);
    result.stitchedImage = StitchImages(srcImage, targetImage, result.rigidTransform);
    if (result.stitchedImage.empty())
    {
        result.success = false;
        result.message = L"Transform estimated, but stitching failed.";
        return;
    }

    if (result.message.IsEmpty())
    {
        result.message.Format(L"%s completed.", result.methodName.GetString());
    }
}
}

CString CRegistrationEngine::GetMethodName(RegistrationMethod method)
{
    switch (method)
    {
    case RegistrationMethod::Phase:
        return L"PHASE";
    case RegistrationMethod::Orb:
        return L"ORB";
    case RegistrationMethod::Akaze:
        return L"AKAZE";
    case RegistrationMethod::Sift:
        return L"SIFT";
    case RegistrationMethod::Ecc:
        return L"ECC";
    default:
        return L"UNKNOWN";
    }
}

RegistrationResult CRegistrationEngine::RegisterAndStitch(
    const cv::Mat& srcImage,
    const cv::Mat& targetImage,
    RegistrationMethod method)
{
    RegistrationResult result;
    result.methodName = GetMethodName(method);

    try
    {
        if (srcImage.empty() || targetImage.empty())
        {
            result.message = L"Please load both source and target images first.";
            return result;
        }

        cv::TickMeter timer;
        timer.start();

        switch (method)
        {
        case RegistrationMethod::Phase:
            result = RunPhaseMethod(srcImage, targetImage);
            break;
        case RegistrationMethod::Orb:
            result = RunFeatureMethod(
                srcImage,
                targetImage,
                L"ORB",
                cv::ORB::create(2500),
                cv::NORM_HAMMING,
                1600,
                0.78f,
                500);
            break;
        case RegistrationMethod::Akaze:
            result = RunFeatureMethod(
                srcImage,
                targetImage,
                L"AKAZE",
                cv::AKAZE::create(),
                cv::NORM_HAMMING,
                1800,
                0.80f,
                700);
            break;
        case RegistrationMethod::Sift:
            result = RunFeatureMethod(
                srcImage,
                targetImage,
                L"SIFT",
                cv::SIFT::create(3000),
                cv::NORM_L2,
                2000,
                0.75f,
                900);
            break;
        case RegistrationMethod::Ecc:
            result = RunEccMethod(srcImage, targetImage);
            break;
        default:
            result.message = L"Unknown registration method.";
            return result;
        }

        FinalizeResult(result, srcImage, targetImage);

        timer.stop();
        result.elapsedMs = timer.getTimeMilli();
    }
    catch (const cv::Exception& ex)
    {
        result.message.Format(L"OpenCV error: %hs", ex.what());
    }
    catch (const std::exception& ex)
    {
        result.message.Format(L"Runtime error: %hs", ex.what());
    }

    return result;
}
