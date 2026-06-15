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

struct FeatureMethodOptions
{
    int maxDimension = 1600;
    float ratioThreshold = 0.78f;
    size_t maxMatchesToKeep = 500;
    size_t visualizationMatches = 80;
    CString accelerationProfile;
    bool usedAcceleration = false;
};

bool IsLargeImagePair(const cv::Mat& srcImage, const cv::Mat& targetImage)
{
    const int maxDimension = std::max(
        std::max(srcImage.cols, srcImage.rows),
        std::max(targetImage.cols, targetImage.rows));
    const double maxPixels = std::max(
        static_cast<double>(srcImage.cols) * static_cast<double>(srcImage.rows),
        static_cast<double>(targetImage.cols) * static_cast<double>(targetImage.rows));
    return maxDimension >= 6000 || maxPixels >= 12000000.0;
}

bool IsVeryLargeImagePair(const cv::Mat& srcImage, const cv::Mat& targetImage)
{
    const int maxDimension = std::max(
        std::max(srcImage.cols, srcImage.rows),
        std::max(targetImage.cols, targetImage.rows));
    const double maxPixels = std::max(
        static_cast<double>(srcImage.cols) * static_cast<double>(srcImage.rows),
        static_cast<double>(targetImage.cols) * static_cast<double>(targetImage.rows));
    return maxDimension >= 10000 || maxPixels >= 30000000.0;
}

double ElapsedMs(int64_t startTick)
{
    return static_cast<double>(cv::getTickCount() - startTick) * 1000.0 / cv::getTickFrequency();
}

void AppendLogLine(CString& logText, const CString& line)
{
    if (!logText.IsEmpty())
    {
        logText += L"\r\n";
    }
    logText += line;
    TRACE(L"[Registration] %s\n", line.GetString());
}

cv::Point2d TransformPoint(const cv::Matx33d& transform, const cv::Point2d& point)
{
    const cv::Vec3d homogeneous(point.x, point.y, 1.0);
    const cv::Vec3d mapped = transform * homogeneous;
    return cv::Point2d(mapped[0], mapped[1]);
}

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

void BuildFeatureVisualization(
    RegistrationResult& result,
    const std::vector<cv::KeyPoint>& srcKeypoints,
    const std::vector<cv::KeyPoint>& targetKeypoints,
    const std::vector<cv::DMatch>& matches,
    const cv::Mat& inlierMask,
    double workingScale,
    size_t maxVisualizationMatches)
{
    if (matches.empty())
    {
        return;
    }

    const cv::Matx33d targetToSrc = result.rigidTransform.inv();
    size_t visualized = 0;
    for (size_t i = 0; i < matches.size(); ++i)
    {
        if (!inlierMask.empty() && inlierMask.at<uchar>(static_cast<int>(i), 0) == 0)
        {
            continue;
        }

        const cv::Point2d srcPoint(
            srcKeypoints[matches[i].queryIdx].pt.x / workingScale,
            srcKeypoints[matches[i].queryIdx].pt.y / workingScale);
        const cv::Point2d targetPoint(
            targetKeypoints[matches[i].trainIdx].pt.x / workingScale,
            targetKeypoints[matches[i].trainIdx].pt.y / workingScale);
        const cv::Point2d mappedSrcOnTarget = TransformPoint(result.rigidTransform, srcPoint);
        const cv::Point2d mappedTargetOnSrc = TransformPoint(targetToSrc, targetPoint);

        // Store only the feature point locations, not the mapped points
        // The mapped points are stored in the lines, and should not be duplicated in visualization points
        result.srcVisualizationPoints.push_back(srcPoint);
        result.srcVisualizationPoints.push_back(mappedTargetOnSrc);  // End point of the line in Source pane
        
        result.targetVisualizationPoints.push_back(targetPoint);
        result.targetVisualizationPoints.push_back(mappedSrcOnTarget);  // End point of the line in Target pane
        
        result.srcVisualizationLines.push_back({srcPoint, mappedTargetOnSrc});
        result.targetVisualizationLines.push_back({mappedSrcOnTarget, targetPoint});

        ++visualized;
        if (visualized >= maxVisualizationMatches)
        {
            break;
        }
    }

    result.hasFeatureVisualization = !result.srcVisualizationLines.empty();
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

    // Blend target image into canvas at the offset position
    const int targetStartX = offsetX;
    const int targetStartY = offsetY;
    const int targetEndX = std::min(targetStartX + targetImage.cols, canvas.cols);
    const int targetEndY = std::min(targetStartY + targetImage.rows, canvas.rows);
    
    if (targetStartX >= 0 && targetStartY >= 0 && targetEndX > targetStartX && targetEndY > targetStartY)
    {
        const int roiWidth = targetEndX - targetStartX;
        const int roiHeight = targetEndY - targetStartY;
        
        cv::Rect targetRoi(targetStartX, targetStartY, roiWidth, roiHeight);
        cv::Rect targetImageRoi(0, 0, roiWidth, roiHeight);
        
        cv::Mat canvasRoi = canvas(targetRoi);
        cv::Mat targetRoiMat = targetImage(targetImageRoi);
        cv::Mat maskRoi = warpedMask(targetRoi);
        
        // Create inverse mask for non-overlapping regions
        cv::Mat nonOverlapMask;
        cv::bitwise_not(maskRoi, nonOverlapMask);
        
        // Copy target where there's no overlap
        targetRoiMat.copyTo(canvasRoi, nonOverlapMask);
        
        // Blend where there's overlap
        cv::Mat blended;
        cv::addWeighted(canvasRoi, 0.5, targetRoiMat, 0.5, 0.0, blended);
        blended.copyTo(canvasRoi, maskRoi);
    }

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
    const FeatureMethodOptions& options)
{
    RegistrationResult result;
    result.methodName = methodName;
    result.accelerationProfile = options.accelerationProfile;
    result.usedAcceleration = options.usedAcceleration;

    {
        CString line;
        line.Format(
            L"[%s] Input src=%dx%d, target=%dx%d",
            methodName.GetString(),
            srcImage.cols,
            srcImage.rows,
            targetImage.cols,
            targetImage.rows);
        AppendLogLine(result.logText, line);
    }
    if (!result.accelerationProfile.IsEmpty())
    {
        CString line;
        line.Format(L"[%s] Acceleration profile: %s", methodName.GetString(), result.accelerationProfile.GetString());
        AppendLogLine(result.logText, line);
    }

    const int64_t prepareTick = cv::getTickCount();
    const WorkingPair pair = PrepareWorkingPair(srcImage, targetImage, options.maxDimension);
    result.workingScale = pair.scale;
    result.workingWidth = pair.src.cols;
    result.workingHeight = pair.src.rows;
    {
        CString line;
        line.Format(
            L"[%s] Prepare working images: %dx%d / %dx%d, scale=%.5f, %.2f ms",
            methodName.GetString(),
            pair.src.cols,
            pair.src.rows,
            pair.target.cols,
            pair.target.rows,
            pair.scale,
            ElapsedMs(prepareTick));
        AppendLogLine(result.logText, line);
    }

    const int64_t grayTick = cv::getTickCount();
    const cv::Mat srcGray = ConvertToGray8(pair.src);
    const cv::Mat targetGray = ConvertToGray8(pair.target);
    {
        CString line;
        line.Format(L"[%s] Convert to gray: %.2f ms", methodName.GetString(), ElapsedMs(grayTick));
        AppendLogLine(result.logText, line);
    }

    const int64_t featureTick = cv::getTickCount();
    std::vector<cv::KeyPoint> srcKeypoints;
    std::vector<cv::KeyPoint> targetKeypoints;
    cv::Mat srcDescriptors;
    cv::Mat targetDescriptors;
    detector->detectAndCompute(srcGray, cv::noArray(), srcKeypoints, srcDescriptors);
    detector->detectAndCompute(targetGray, cv::noArray(), targetKeypoints, targetDescriptors);

    result.keypointsSrc = static_cast<int>(srcKeypoints.size());
    result.keypointsTarget = static_cast<int>(targetKeypoints.size());
    {
        CString line;
        line.Format(
            L"[%s] Detect features: src=%d, target=%d, %.2f ms",
            methodName.GetString(),
            result.keypointsSrc,
            result.keypointsTarget,
            ElapsedMs(featureTick));
        AppendLogLine(result.logText, line);
    }
    if (srcDescriptors.empty() || targetDescriptors.empty())
    {
        result.message = L"Not enough features detected.";
        return result;
    }

    const int64_t matchTick = cv::getTickCount();
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

        if (candidates[0].distance < options.ratioThreshold * candidates[1].distance)
        {
            goodMatches.push_back(candidates[0]);
        }
    }

    std::sort(goodMatches.begin(), goodMatches.end(), [](const cv::DMatch& left, const cv::DMatch& right)
    {
        return left.distance < right.distance;
    });
    if (goodMatches.size() > options.maxMatchesToKeep)
    {
        goodMatches.resize(options.maxMatchesToKeep);
    }

    result.matches = static_cast<int>(goodMatches.size());
    {
        CString line;
        line.Format(
            L"[%s] Match + ratio test: kept=%d, %.2f ms",
            methodName.GetString(),
            result.matches,
            ElapsedMs(matchTick));
        AppendLogLine(result.logText, line);
    }
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

    const int64_t ransacTick = cv::getTickCount();
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
    BuildFeatureVisualization(
        result,
        srcKeypoints,
        targetKeypoints,
        goodMatches,
        inlierMask,
        pair.scale,
        options.visualizationMatches);
    {
        CString line;
        line.Format(
            L"[%s] RANSAC estimate: inliers=%d/%d, %.2f ms",
            methodName.GetString(),
            result.inliers,
            result.matches,
            ElapsedMs(ransacTick));
        AppendLogLine(result.logText, line);
    }
    if (result.hasFeatureVisualization)
    {
        CString line;
        line.Format(
            L"[%s] Visualization prepared for %zu inlier correspondences.",
            methodName.GetString(),
            result.srcVisualizationLines.size());
        AppendLogLine(result.logText, line);
    }
    result.success = true;
    return result;
}

RegistrationResult RunPhaseMethod(const cv::Mat& srcImage, const cv::Mat& targetImage)
{
    RegistrationResult result;
    result.methodName = L"PHASE";
    const bool largeImage = IsLargeImagePair(srcImage, targetImage);
    const bool veryLargeImage = IsVeryLargeImagePair(srcImage, targetImage);
    const int maxDimension = veryLargeImage ? 1200 : (largeImage ? 1400 : 1600);
    result.usedAcceleration = largeImage;
    result.accelerationProfile = largeImage
        ? L"Large-image mode: stronger downsample for fast translation estimation."
        : L"Standard mode.";
    {
        CString line;
        line.Format(L"[PHASE] Input src=%dx%d, target=%dx%d", srcImage.cols, srcImage.rows, targetImage.cols, targetImage.rows);
        AppendLogLine(result.logText, line);
    }
    AppendLogLine(result.logText, CString(L"[PHASE] Acceleration profile: ") + result.accelerationProfile);

    const int64_t prepareTick = cv::getTickCount();
    const WorkingPair pair = PrepareWorkingPair(srcImage, targetImage, maxDimension);
    result.workingScale = pair.scale;
    result.workingWidth = pair.src.cols;
    result.workingHeight = pair.src.rows;
    {
        CString line;
        line.Format(
            L"[PHASE] Prepare working images: %dx%d / %dx%d, scale=%.5f, %.2f ms",
            pair.src.cols,
            pair.src.rows,
            pair.target.cols,
            pair.target.rows,
            pair.scale,
            ElapsedMs(prepareTick));
        AppendLogLine(result.logText, line);
    }

    const int64_t grayTick = cv::getTickCount();
    cv::Mat srcGray;
    cv::Mat targetGray;
    PadToSameSize(ConvertToGray8(pair.src), ConvertToGray8(pair.target), srcGray, targetGray);
    {
        CString line;
        line.Format(L"[PHASE] Convert + pad grayscale: %.2f ms", ElapsedMs(grayTick));
        AppendLogLine(result.logText, line);
    }

    const int64_t floatTick = cv::getTickCount();
    cv::Mat srcFloat;
    cv::Mat targetFloat;
    srcGray.convertTo(srcFloat, CV_32F);
    targetGray.convertTo(targetFloat, CV_32F);
    {
        CString line;
        line.Format(L"[PHASE] Convert to float: %.2f ms", ElapsedMs(floatTick));
        AppendLogLine(result.logText, line);
    }

    const int64_t phaseTick = cv::getTickCount();
    cv::Mat window;
    cv::createHanningWindow(window, srcFloat.size(), CV_32F);
    const cv::Point2d shift = cv::phaseCorrelate(srcFloat, targetFloat, window);
    {
        CString line;
        line.Format(L"[PHASE] Phase correlation shift=(%.3f, %.3f), %.2f ms", shift.x, shift.y, ElapsedMs(phaseTick));
        AppendLogLine(result.logText, line);
    }

    const int64_t rmseTick = cv::getTickCount();
    const cv::Matx33d forward = BuildTranslationTransform(shift.x, shift.y);
    const cv::Matx33d backward = BuildTranslationTransform(-shift.x, -shift.y);
    const double forwardRmse = ComputeImageRmse(pair.src, pair.target, forward);
    const double backwardRmse = ComputeImageRmse(pair.src, pair.target, backward);
    {
        CString line;
        line.Format(
            L"[PHASE] Direction check: forward RMSE=%.4f, backward RMSE=%.4f, %.2f ms",
            forwardRmse,
            backwardRmse,
            ElapsedMs(rmseTick));
        AppendLogLine(result.logText, line);
    }

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
    const bool largeImage = IsLargeImagePair(srcImage, targetImage);
    const bool veryLargeImage = IsVeryLargeImagePair(srcImage, targetImage);
    const int maxDimension = veryLargeImage ? 1200 : (largeImage ? 1500 : 1800);
    const int maxLevel = veryLargeImage ? 2 : 2;
    const int maxIterations = veryLargeImage ? 60 : (largeImage ? 80 : 100);
    result.usedAcceleration = largeImage;
    result.accelerationProfile = largeImage
        ? L"Large-image mode: reduced ECC resolution and iterations with phase-based initialization."
        : L"Standard mode.";
    {
        CString line;
        line.Format(L"[ECC] Input src=%dx%d, target=%dx%d", srcImage.cols, srcImage.rows, targetImage.cols, targetImage.rows);
        AppendLogLine(result.logText, line);
    }
    AppendLogLine(result.logText, CString(L"[ECC] Acceleration profile: ") + result.accelerationProfile);

    const int64_t prepareTick = cv::getTickCount();
    const WorkingPair pair = PrepareWorkingPair(srcImage, targetImage, maxDimension);
    result.workingScale = pair.scale;
    result.workingWidth = pair.src.cols;
    result.workingHeight = pair.src.rows;
    {
        CString line;
        line.Format(
            L"[ECC] Prepare working images: %dx%d / %dx%d, scale=%.5f, %.2f ms",
            pair.src.cols,
            pair.src.rows,
            pair.target.cols,
            pair.target.rows,
            pair.scale,
            ElapsedMs(prepareTick));
        AppendLogLine(result.logText, line);
    }

    const int64_t grayTick = cv::getTickCount();
    cv::Mat srcGray;
    cv::Mat targetGray;
    PadToSameSize(ConvertToGray8(pair.src), ConvertToGray8(pair.target), srcGray, targetGray);
    {
        CString line;
        line.Format(L"[ECC] Convert + pad grayscale: %.2f ms", ElapsedMs(grayTick));
        AppendLogLine(result.logText, line);
    }

    const int64_t floatTick = cv::getTickCount();
    cv::Mat srcFloat;
    cv::Mat targetFloat;
    srcGray.convertTo(srcFloat, CV_32F, 1.0 / 255.0);
    targetGray.convertTo(targetFloat, CV_32F, 1.0 / 255.0);
    {
        CString line;
        line.Format(L"[ECC] Normalize to float: %.2f ms", ElapsedMs(floatTick));
        AppendLogLine(result.logText, line);
    }

    const int64_t pyramidTick = cv::getTickCount();
    std::vector<cv::Mat> srcPyramid;
    std::vector<cv::Mat> targetPyramid;
    cv::buildPyramid(srcFloat, srcPyramid, maxLevel);
    cv::buildPyramid(targetFloat, targetPyramid, maxLevel);
    {
        CString line;
        line.Format(
            L"[ECC] Build pyramid: levels=%d, coarsest=%dx%d, %.2f ms",
            maxLevel + 1,
            srcPyramid.back().cols,
            srcPyramid.back().rows,
            ElapsedMs(pyramidTick));
        AppendLogLine(result.logText, line);
    }

    cv::Mat warp = cv::Mat::eye(2, 3, CV_32F);

    {
        const int64_t initTick = cv::getTickCount();
        cv::Mat coarseWindow;
        cv::createHanningWindow(coarseWindow, srcPyramid.back().size(), CV_32F);
        const cv::Point2d initialShift = cv::phaseCorrelate(srcPyramid.back(), targetPyramid.back(), coarseWindow);
        warp.at<float>(0, 2) = static_cast<float>(initialShift.x);
        warp.at<float>(1, 2) = static_cast<float>(initialShift.y);
        CString line;
        line.Format(
            L"[ECC] Phase init at coarsest level: shift=(%.3f, %.3f), %.2f ms",
            initialShift.x,
            initialShift.y,
            ElapsedMs(initTick));
        AppendLogLine(result.logText, line);
    }

    const int64_t eccTick = cv::getTickCount();
    for (int level = maxLevel; level >= 0; --level)
    {
        if (level < maxLevel)
        {
            warp.at<float>(0, 2) *= 2.0f;
            warp.at<float>(1, 2) *= 2.0f;
        }

        const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, maxIterations, 1e-5);
        cv::findTransformECC(
            targetPyramid[level],
            srcPyramid[level],
            warp,
            cv::MOTION_EUCLIDEAN,
            criteria,
            cv::noArray(),
            5);
    }
    {
        CString line;
        line.Format(
            L"[ECC] Optimize transform across pyramid: iterations<=%d, %.2f ms",
            maxIterations,
            ElapsedMs(eccTick));
        AppendLogLine(result.logText, line);
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

    const int64_t rmseTick = cv::getTickCount();
    result.rmse = ComputeImageRmse(srcImage, targetImage, result.rigidTransform);
    {
        CString line;
        line.Format(L"[%s] Compute overlap RMSE: %.4f, %.2f ms", result.methodName.GetString(), result.rmse, ElapsedMs(rmseTick));
        AppendLogLine(result.logText, line);
    }
    const int64_t stitchTick = cv::getTickCount();
    result.stitchedImage = StitchImages(srcImage, targetImage, result.rigidTransform);
    if (result.stitchedImage.empty())
    {
        result.success = false;
        result.message = L"Transform estimated, but stitching failed.";
        return;
    }
    {
        CString line;
        line.Format(
            L"[%s] Stitch images: result=%dx%d, %.2f ms",
            result.methodName.GetString(),
            result.stitchedImage.cols,
            result.stitchedImage.rows,
            ElapsedMs(stitchTick));
        AppendLogLine(result.logText, line);
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

        const bool largeImage = IsLargeImagePair(srcImage, targetImage);
        const bool veryLargeImage = IsVeryLargeImagePair(srcImage, targetImage);
        cv::TickMeter timer;
        timer.start();

        switch (method)
        {
        case RegistrationMethod::Phase:
            result = RunPhaseMethod(srcImage, targetImage);
            break;
        case RegistrationMethod::Orb:
        {
            FeatureMethodOptions options;
            options.maxDimension = veryLargeImage ? 1200 : (largeImage ? 1400 : 1600);
            options.ratioThreshold = 0.78f;
            options.maxMatchesToKeep = veryLargeImage ? 260 : (largeImage ? 360 : 500);
            options.visualizationMatches = 80;
            options.usedAcceleration = largeImage;
            options.accelerationProfile = largeImage
                ? L"Large-image mode: ORB features capped and stronger downsample enabled."
                : L"Standard mode.";
            result = RunFeatureMethod(
                srcImage,
                targetImage,
                L"ORB",
                cv::ORB::create(veryLargeImage ? 1400 : (largeImage ? 1800 : 2500)),
                cv::NORM_HAMMING,
                options);
            break;
        }
        case RegistrationMethod::Akaze:
        {
            FeatureMethodOptions options;
            options.maxDimension = veryLargeImage ? 1200 : (largeImage ? 1500 : 1800);
            options.ratioThreshold = 0.80f;
            options.maxMatchesToKeep = veryLargeImage ? 280 : (largeImage ? 420 : 700);
            options.visualizationMatches = 80;
            options.usedAcceleration = largeImage;
            options.accelerationProfile = largeImage
                ? L"Large-image mode: AKAZE uses reduced working resolution and fewer retained matches."
                : L"Standard mode.";
            result = RunFeatureMethod(
                srcImage,
                targetImage,
                L"AKAZE",
                cv::AKAZE::create(),
                cv::NORM_HAMMING,
                options);
            break;
        }
        case RegistrationMethod::Sift:
        {
            FeatureMethodOptions options;
            options.maxDimension = veryLargeImage ? 1280 : (largeImage ? 1500 : 2000);
            options.ratioThreshold = 0.75f;
            options.maxMatchesToKeep = veryLargeImage ? 320 : (largeImage ? 500 : 900);
            options.visualizationMatches = 80;
            options.usedAcceleration = largeImage;
            options.accelerationProfile = largeImage
                ? L"Large-image mode: SIFT feature count capped and working resolution reduced."
                : L"Standard mode.";
            result = RunFeatureMethod(
                srcImage,
                targetImage,
                L"SIFT",
                cv::SIFT::create(veryLargeImage ? 1400 : (largeImage ? 1800 : 3000)),
                cv::NORM_L2,
                options);
            break;
        }
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
        {
            CString line;
            line.Format(L"[%s] Total elapsed: %.2f ms", result.methodName.GetString(), result.elapsedMs);
            AppendLogLine(result.logText, line);
        }
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
