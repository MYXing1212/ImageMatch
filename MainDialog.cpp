#include "framework.h"
#include "MainDialog.h"
#include "RegistrationEngine.h"
#include "resource.h"

namespace
{
cv::Mat ReadImageFromFile(const CString& filePath)
{
    FILE* file = nullptr;
    if (_wfopen_s(&file, filePath, L"rb") != 0 || file == nullptr)
    {
        return {};
    }

    _fseeki64(file, 0, SEEK_END);
    const auto fileSize = _ftelli64(file);
    if (fileSize <= 0)
    {
        fclose(file);
        return {};
    }

    std::vector<uchar> buffer(static_cast<size_t>(fileSize));
    rewind(file);
    const size_t bytesRead = fread(buffer.data(), 1, buffer.size(), file);
    fclose(file);
    if (bytesRead != buffer.size())
    {
        return {};
    }

    return cv::imdecode(buffer, cv::IMREAD_COLOR);
}

CString FormatMatrix(const cv::Matx33d& matrix)
{
    std::wostringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(6);
    for (int row = 0; row < 3; ++row)
    {
        stream << L"["
               << matrix(row, 0) << L", "
               << matrix(row, 1) << L", "
               << matrix(row, 2) << L"]";
        if (row < 2)
        {
            stream << L"\r\n";
        }
    }
    return CString(stream.str().c_str());
}

void ApplyVisualizationOverlay(
    CImageViewPane& pane,
    const std::vector<cv::Point2d>& keypoints,
    const std::vector<MatchLineSegment>& lineSegments)
{
    std::vector<ImageOverlayPoint> points;
    std::vector<ImageOverlayLine> lines;
    points.reserve(keypoints.size() + lineSegments.size());
    lines.reserve(lineSegments.size());

    for (const auto& point : keypoints)
    {
        points.push_back({point, RGB(80, 255, 80), 3});
    }

    for (const auto& segment : lineSegments)
    {
        lines.push_back({segment.start, segment.end, RGB(0, 220, 255), 1});
        points.push_back({segment.end, RGB(255, 180, 0), 2});
    }

    pane.SetOverlay(points, lines);
}
}

BEGIN_MESSAGE_MAP(CMainDialog, CDialogEx)
    ON_BN_CLICKED(IDC_BUTTON_LOAD_SRC, &CMainDialog::OnBnClickedLoadSrc)
    ON_BN_CLICKED(IDC_BUTTON_LOAD_TARGET, &CMainDialog::OnBnClickedLoadTarget)
    ON_COMMAND_RANGE(IDC_BUTTON_METHOD_PHASE, IDC_BUTTON_METHOD_ECC, &CMainDialog::OnBnClickedMethodRange)
    ON_BN_CLICKED(IDC_BUTTON_RESET, &CMainDialog::OnBnClickedReset)
    ON_WM_SIZE()
    ON_MESSAGE(WM_IMAGE_PANE_FILE_DROPPED, &CMainDialog::OnImagePaneFileDropped)
END_MESSAGE_MAP()

CMainDialog::CMainDialog()
    : CDialogEx(IDD_IMAGE_MATCHING_DIALOG)
{
}

void CMainDialog::DoDataExchange(CDataExchange* pDX)
{
    CDialogEx::DoDataExchange(pDX);
}

BOOL CMainDialog::OnInitDialog()
{
    CDialogEx::OnInitDialog();

    m_resultEdit.SubclassDlgItem(IDC_EDIT_RESULT, this);
    m_resultEdit.SetWindowTextW(L"Load or drop source and target images, then click a method button.");

    m_srcPane.CreatePane(this, IDC_PANE_SRC, L"Source");
    m_targetPane.CreatePane(this, IDC_PANE_TARGET, L"Target");
    m_resultPane.CreatePane(this, IDC_PANE_RESULT, L"Stitched");

    RepositionLayout();
    return TRUE;
}

void CMainDialog::OnBnClickedLoadSrc()
{
    LoadImage(true);
}

void CMainDialog::OnBnClickedLoadTarget()
{
    LoadImage(false);
}

void CMainDialog::OnBnClickedMethodRange(UINT controlId)
{
    RunRegistration(ResolveMethodFromButton(controlId));
}

void CMainDialog::OnBnClickedReset()
{
    m_srcPane.ResetView();
    m_targetPane.ResetView();
    m_resultPane.ResetView();
}

void CMainDialog::OnSize(UINT nType, int cx, int cy)
{
    CDialogEx::OnSize(nType, cx, cy);
    RepositionLayout();
}

bool CMainDialog::LoadImage(bool isSourceImage)
{
    const CString dialogTitle = isSourceImage ? L"Select source image" : L"Select target image";
    CFileDialog fileDialog(
        TRUE,
        nullptr,
        nullptr,
        OFN_FILEMUSTEXIST | OFN_HIDEREADONLY,
        L"Image Files (*.bmp;*.jpg;*.jpeg;*.png;*.tif;*.tiff)|*.bmp;*.jpg;*.jpeg;*.png;*.tif;*.tiff|All Files (*.*)|*.*||",
        this);
    fileDialog.m_ofn.lpstrTitle = dialogTitle;

    if (fileDialog.DoModal() != IDOK)
    {
        return false;
    }

    return LoadImageFromPath(fileDialog.GetPathName(), isSourceImage);
}

bool CMainDialog::LoadImageFromPath(const CString& filePath, bool isSourceImage)
{
    cv::Mat image = ReadImageFromFile(filePath);
    if (image.empty())
    {
        CString errorText;
        errorText.Format(L"Failed to read image: %s", filePath.GetString());
        UpdateResultText(errorText);
        return false;
    }

    if (isSourceImage)
    {
        m_srcImage = image;
        m_srcPane.SetImage(m_srcImage);
        m_srcPane.ClearOverlay();
    }
    else
    {
        m_targetImage = image;
        m_targetPane.SetImage(m_targetImage);
    }

    m_srcPane.ClearOverlay();
    m_targetPane.ClearOverlay();
    m_stitchedImage.release();
    m_resultPane.ClearImage();
    UpdateResultText(FormatRegistrationSummary());
    return true;
}

void CMainDialog::RunRegistration(RegistrationMethod method)
{
    if (m_srcImage.empty() || m_targetImage.empty())
    {
        UpdateResultText(L"Please load both source and target images first.");
        return;
    }

    CString runningText;
    runningText.Format(L"Running %s ...", CRegistrationEngine::GetMethodName(method).GetString());
    UpdateResultText(runningText);

    const RegistrationResult result = CRegistrationEngine::RegisterAndStitch(m_srcImage, m_targetImage, method);
    if (!result.success)
    {
        m_resultPane.ShowToast(L"Registration Failed", RGB(255, 0, 0), 1800);
        m_srcPane.SetImage(m_srcImage);
        m_targetPane.SetImage(m_targetImage);
        m_srcPane.ClearOverlay();
        m_targetPane.ClearOverlay();
        CString errorText;
        errorText.Format(
            L"%s failed.\r\n%s%s%s",
            result.methodName.GetString(),
            result.message.GetString(),
            result.logText.IsEmpty() ? L"" : L"\r\n\r\nLogs:\r\n",
            result.logText.GetString());
        UpdateResultText(errorText);
        return;
    }

    m_srcPane.SetImage(m_srcImage);
    m_targetPane.SetImage(m_targetImage);
    if (result.hasFeatureVisualization)
    {
        ApplyVisualizationOverlay(m_srcPane, result.srcVisualizationPoints, result.srcVisualizationLines);
        ApplyVisualizationOverlay(m_targetPane, result.targetVisualizationPoints, result.targetVisualizationLines);
    }
    else
    {
        m_srcPane.ClearOverlay();
        m_targetPane.ClearOverlay();
    }

    m_stitchedImage = result.stitchedImage.clone();
    m_resultPane.SetImage(m_stitchedImage);
    m_resultPane.ClearOverlay();
    m_resultPane.ShowToast(L"Registration Succeeded", RGB(0, 200, 0), 1800);
    UpdateResultText(BuildResultSummary(result));
}

void CMainDialog::UpdateResultText(const CString& text)
{
    m_resultText = text;
    if (m_resultEdit.GetSafeHwnd() != nullptr)
    {
        m_resultEdit.SetWindowTextW(m_resultText);
    }
}

void CMainDialog::RepositionLayout()
{
    if (GetSafeHwnd() == nullptr)
    {
        return;
    }

    CRect clientRect;
    GetClientRect(&clientRect);
    if (clientRect.Width() <= 0 || clientRect.Height() <= 0)
    {
        return;
    }

    constexpr int kMargin = 8;
    constexpr int kGap = 8;
    constexpr int kButtonHeight = 28;
    constexpr int kResultHeight = 84;
    int x = kMargin;
    const int y = kMargin;

    struct ButtonLayout
    {
        UINT id;
        int width;
    };

    const std::array<ButtonLayout, 8> buttons = {{
        {IDC_BUTTON_LOAD_SRC, 88},
        {IDC_BUTTON_LOAD_TARGET, 96},
        {IDC_BUTTON_METHOD_PHASE, 74},
        {IDC_BUTTON_METHOD_ORB, 68},
        {IDC_BUTTON_METHOD_AKAZE, 78},
        {IDC_BUTTON_METHOD_SIFT, 68},
        {IDC_BUTTON_METHOD_ECC, 68},
        {IDC_BUTTON_RESET, 92},
    }};

    for (const auto& buttonInfo : buttons)
    {
        if (CWnd* button = GetDlgItem(buttonInfo.id))
        {
            button->MoveWindow(x, y, buttonInfo.width, kButtonHeight);
            x += buttonInfo.width + kGap;
        }
    }

    const int resultTop = y + kButtonHeight + kGap;
    const int resultWidth = std::max(200, clientRect.Width() - 2 * kMargin);
    if (m_resultEdit.GetSafeHwnd() != nullptr)
    {
        m_resultEdit.MoveWindow(kMargin, resultTop, resultWidth, kResultHeight);
    }

    const int panesTop = resultTop + kResultHeight + kGap;
    const int panesHeight = std::max(200, clientRect.Height() - panesTop - kMargin);
    const int topPaneHeight = std::max(120, (panesHeight - kGap) / 2);
    const int paneWidth = std::max(120, (clientRect.Width() - 2 * kMargin - kGap) / 2);
    const int rightPaneX = kMargin + paneWidth + kGap;

    if (m_srcPane.GetSafeHwnd() != nullptr)
    {
        m_srcPane.MoveWindow(kMargin, panesTop, paneWidth, topPaneHeight);
    }
    if (m_targetPane.GetSafeHwnd() != nullptr)
    {
        m_targetPane.MoveWindow(rightPaneX, panesTop, paneWidth, topPaneHeight);
    }
    if (m_resultPane.GetSafeHwnd() != nullptr)
    {
        m_resultPane.MoveWindow(
            kMargin,
            panesTop + topPaneHeight + kGap,
            clientRect.Width() - 2 * kMargin,
            std::max(120, panesHeight - topPaneHeight - kGap));
    }
}

CString CMainDialog::FormatRegistrationSummary() const
{
    CString summary;
    summary.Format(
        L"Status:\r\n"
        L"Source image: %s\r\n"
        L"Target image: %s\r\n"
        L"Tips: left-drag to pan, mouse wheel to zoom around cursor, and drop files into Source or Target pane.",
        m_srcImage.empty() ? L"not loaded" : L"loaded",
        m_targetImage.empty() ? L"not loaded" : L"loaded");
    return summary;
}

CString CMainDialog::BuildResultSummary(const RegistrationResult& result) const
{
    CString summary;
    summary.Format(
        L"Method: %s\r\n"
        L"Status: %s\r\n"
        L"Elapsed: %.2f ms\r\n"
        L"Working scale: %.4f\r\n"
        L"Working size: %d x %d\r\n"
        L"Acceleration: %s\r\n"
        L"Source keypoints: %d\r\n"
        L"Target keypoints: %d\r\n"
        L"Good matches: %d\r\n"
        L"Inliers: %d\r\n"
        L"Rotation (deg): %.4f\r\n"
        L"Translation (tx, ty): (%.4f, %.4f)\r\n"
        L"Overlap RMSE: %.4f\r\n"
        L"Feature overlay: %s\r\n"
        L"RT matrix:\r\n%s%s%s",
        result.methodName.GetString(),
        result.message.GetString(),
        result.elapsedMs,
        result.workingScale,
        result.workingWidth,
        result.workingHeight,
        result.accelerationProfile.IsEmpty() ? L"n/a" : result.accelerationProfile.GetString(),
        result.keypointsSrc,
        result.keypointsTarget,
        result.matches,
        result.inliers,
        result.angleDegrees,
        result.translation.x,
        result.translation.y,
        result.rmse,
        result.hasFeatureVisualization ? L"shown in Source/Target panes" : L"not applicable",
        FormatMatrix(result.rigidTransform).GetString(),
        result.logText.IsEmpty() ? L"" : L"\r\n\r\nLogs:\r\n",
        result.logText.GetString());
    return summary;
}

RegistrationMethod CMainDialog::ResolveMethodFromButton(UINT controlId) const
{
    switch (controlId)
    {
    case IDC_BUTTON_METHOD_PHASE:
        return RegistrationMethod::Phase;
    case IDC_BUTTON_METHOD_ORB:
        return RegistrationMethod::Orb;
    case IDC_BUTTON_METHOD_AKAZE:
        return RegistrationMethod::Akaze;
    case IDC_BUTTON_METHOD_SIFT:
        return RegistrationMethod::Sift;
    case IDC_BUTTON_METHOD_ECC:
        return RegistrationMethod::Ecc;
    default:
        return RegistrationMethod::Orb;
    }
}

LRESULT CMainDialog::OnImagePaneFileDropped(WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(wParam);

    std::unique_ptr<ImagePaneDropPayload> payload(reinterpret_cast<ImagePaneDropPayload*>(lParam));
    if (!payload)
    {
        return 0;
    }

    if (payload->paneId == IDC_PANE_SRC)
    {
        LoadImageFromPath(payload->filePath, true);
    }
    else if (payload->paneId == IDC_PANE_TARGET)
    {
        LoadImageFromPath(payload->filePath, false);
    }

    return 0;
}
