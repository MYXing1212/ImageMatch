#include "framework.h"
#include "ImageViewPane.h"

namespace
{
constexpr double kMinScale = 0.05;
constexpr double kMaxScale = 40.0;
}

BEGIN_MESSAGE_MAP(CImageViewPane, CWnd)
    ON_WM_PAINT()
    ON_WM_ERASEBKGND()
    ON_WM_SIZE()
    ON_WM_LBUTTONDOWN()
    ON_WM_LBUTTONUP()
    ON_WM_MOUSEMOVE()
    ON_WM_MOUSEWHEEL()
    ON_WM_DROPFILES()
END_MESSAGE_MAP()

CImageViewPane::CImageViewPane()
    : m_scale(1.0),
      m_offset(0.0, 0.0),
      m_isDragging(false),
      m_lastMousePoint(0, 0)
{
}

BOOL CImageViewPane::CreatePane(CWnd* pParentWnd, UINT controlId, const CString& title)
{
    m_title = title;
    const CString className = AfxRegisterWndClass(
        CS_DBLCLKS,
        ::LoadCursor(nullptr, IDC_ARROW),
        reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1),
        nullptr);

    const BOOL created = CreateEx(
        WS_EX_CLIENTEDGE,
        className,
        title,
        WS_CHILD | WS_VISIBLE | WS_TABSTOP,
        CRect(0, 0, 0, 0),
        pParentWnd,
        controlId);

    if (created)
    {
        DragAcceptFiles(TRUE);
    }
    return created;
}

void CImageViewPane::SetImage(const cv::Mat& image)
{
    m_image = NormalizeToBgr(image);
    FitImageToClient();
    Invalidate();
}

void CImageViewPane::ClearImage()
{
    m_image.release();
    m_scale = 1.0;
    m_offset = cv::Point2d(0.0, 0.0);
    Invalidate();
}

void CImageViewPane::ResetView()
{
    FitImageToClient();
    Invalidate();
}

void CImageViewPane::OnPaint()
{
    CPaintDC dc(this);
    CRect clientRect;
    GetClientRect(&clientRect);
    if (clientRect.Width() <= 0 || clientRect.Height() <= 0)
    {
        return;
    }

    CDC memoryDc;
    memoryDc.CreateCompatibleDC(&dc);

    CBitmap bitmap;
    bitmap.CreateCompatibleBitmap(&dc, clientRect.Width(), clientRect.Height());
    CBitmap* oldBitmap = memoryDc.SelectObject(&bitmap);

    DrawContent(memoryDc, clientRect);
    dc.BitBlt(0, 0, clientRect.Width(), clientRect.Height(), &memoryDc, 0, 0, SRCCOPY);

    memoryDc.SelectObject(oldBitmap);
}

void CImageViewPane::DrawContent(CDC& dc, const CRect& clientRect)
{
    CRect textRect = clientRect;
    dc.FillSolidRect(clientRect, RGB(30, 30, 30));
    dc.SetBkMode(TRANSPARENT);
    dc.SetTextColor(RGB(220, 220, 220));
    dc.DrawText(m_title, textRect, DT_LEFT | DT_TOP | DT_SINGLELINE | DT_NOPREFIX);

    if (m_image.empty())
    {
        CRect messageRect = clientRect;
        CString message;
        message.Format(L"%s: drop image here or click Load", m_title.GetString());
        dc.DrawText(message, messageRect, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX);
        return;
    }

    const int destX = static_cast<int>(std::lround(m_offset.x));
    const int destY = static_cast<int>(std::lround(m_offset.y));
    const int destWidth = std::max(1, static_cast<int>(std::lround(m_image.cols * m_scale)));
    const int destHeight = std::max(1, static_cast<int>(std::lround(m_image.rows * m_scale)));

    BITMAPINFO bitmapInfo = {};
    bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapInfo.bmiHeader.biWidth = m_image.cols;
    bitmapInfo.bmiHeader.biHeight = -m_image.rows;
    bitmapInfo.bmiHeader.biPlanes = 1;
    bitmapInfo.bmiHeader.biBitCount = 24;
    bitmapInfo.bmiHeader.biCompression = BI_RGB;

    dc.SetStretchBltMode(HALFTONE);
    ::SetBrushOrgEx(dc.GetSafeHdc(), 0, 0, nullptr);

    ::StretchDIBits(
        dc.GetSafeHdc(),
        destX,
        destY,
        destWidth,
        destHeight,
        0,
        0,
        m_image.cols,
        m_image.rows,
        m_image.data,
        &bitmapInfo,
        DIB_RGB_COLORS,
        SRCCOPY);
}

BOOL CImageViewPane::OnEraseBkgnd(CDC* pDC)
{
    UNREFERENCED_PARAMETER(pDC);
    return TRUE;
}

void CImageViewPane::OnSize(UINT nType, int cx, int cy)
{
    CWnd::OnSize(nType, cx, cy);
    if (!m_image.empty() && cx > 0 && cy > 0 && m_scale <= kMinScale)
    {
        FitImageToClient();
    }
}

void CImageViewPane::OnLButtonDown(UINT nFlags, CPoint point)
{
    SetFocus();
    SetCapture();
    m_isDragging = true;
    m_lastMousePoint = point;
    CWnd::OnLButtonDown(nFlags, point);
}

void CImageViewPane::OnLButtonUp(UINT nFlags, CPoint point)
{
    if (m_isDragging)
    {
        ReleaseCapture();
        m_isDragging = false;
    }
    CWnd::OnLButtonUp(nFlags, point);
}

void CImageViewPane::OnMouseMove(UINT nFlags, CPoint point)
{
    if (m_isDragging && (nFlags & MK_LBUTTON) != 0)
    {
        const CPoint delta = point - m_lastMousePoint;
        m_offset.x += delta.x;
        m_offset.y += delta.y;
        m_lastMousePoint = point;
        RedrawWindow(nullptr, nullptr, RDW_INVALIDATE | RDW_UPDATENOW | RDW_NOERASE);
    }
    CWnd::OnMouseMove(nFlags, point);
}

BOOL CImageViewPane::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
    if (m_image.empty())
    {
        return CWnd::OnMouseWheel(nFlags, zDelta, pt);
    }

    ScreenToClient(&pt);
    const cv::Point2d imagePoint = ClientToImage(pt);

    const double factor = (zDelta > 0) ? 1.15 : (1.0 / 1.15);
    const double newScale = std::clamp(m_scale * factor, kMinScale, kMaxScale);
    if (std::abs(newScale - m_scale) < 1e-8)
    {
        return TRUE;
    }

    m_scale = newScale;
    m_offset.x = pt.x - imagePoint.x * m_scale;
    m_offset.y = pt.y - imagePoint.y * m_scale;
    RedrawWindow(nullptr, nullptr, RDW_INVALIDATE | RDW_UPDATENOW | RDW_NOERASE);
    return TRUE;
}

void CImageViewPane::OnDropFiles(HDROP hDropInfo)
{
    wchar_t buffer[MAX_PATH] = {};
    const UINT fileCount = DragQueryFileW(hDropInfo, 0xFFFFFFFF, nullptr, 0);
    if (fileCount > 0)
    {
        DragQueryFileW(hDropInfo, 0, buffer, MAX_PATH);
        auto* payload = new ImagePaneDropPayload();
        payload->paneId = GetDlgCtrlID();
        payload->filePath = buffer;
        GetParent()->PostMessage(WM_IMAGE_PANE_FILE_DROPPED, 0, reinterpret_cast<LPARAM>(payload));
    }

    DragFinish(hDropInfo);
}

void CImageViewPane::FitImageToClient()
{
    if (m_image.empty())
    {
        return;
    }

    CRect clientRect;
    GetClientRect(&clientRect);
    if (clientRect.Width() <= 0 || clientRect.Height() <= 0)
    {
        return;
    }

    const double scaleX = static_cast<double>(clientRect.Width()) / static_cast<double>(m_image.cols);
    const double scaleY = static_cast<double>(clientRect.Height()) / static_cast<double>(m_image.rows);
    m_scale = std::clamp(std::min(scaleX, scaleY), kMinScale, kMaxScale);
    m_offset.x = (clientRect.Width() - m_image.cols * m_scale) * 0.5;
    m_offset.y = (clientRect.Height() - m_image.rows * m_scale) * 0.5;
}

cv::Point2d CImageViewPane::ClientToImage(const CPoint& point) const
{
    return cv::Point2d(
        (point.x - m_offset.x) / m_scale,
        (point.y - m_offset.y) / m_scale);
}

cv::Mat CImageViewPane::NormalizeToBgr(const cv::Mat& image) const
{
    if (image.empty())
    {
        return {};
    }

    cv::Mat result;
    if (image.type() == CV_8UC3)
    {
        result = image.clone();
    }
    else if (image.type() == CV_8UC1)
    {
        cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
    }
    else if (image.type() == CV_8UC4)
    {
        cv::cvtColor(image, result, cv::COLOR_BGRA2BGR);
    }
    else
    {
        cv::Mat normalized;
        image.convertTo(normalized, CV_8U);
        if (normalized.channels() == 1)
        {
            cv::cvtColor(normalized, result, cv::COLOR_GRAY2BGR);
        }
        else if (normalized.channels() == 4)
        {
            cv::cvtColor(normalized, result, cv::COLOR_BGRA2BGR);
        }
        else
        {
            result = normalized;
        }
    }
    return result;
}
