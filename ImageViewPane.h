#pragma once

#include "framework.h"

constexpr UINT WM_IMAGE_PANE_FILE_DROPPED = WM_APP + 101;

struct ImagePaneDropPayload
{
    UINT paneId = 0;
    CString filePath;
};

struct ImageOverlayPoint
{
    cv::Point2d position = cv::Point2d(0.0, 0.0);
    COLORREF color = RGB(0, 255, 0);
    int radius = 3;
};

struct ImageOverlayLine
{
    cv::Point2d start = cv::Point2d(0.0, 0.0);
    cv::Point2d end = cv::Point2d(0.0, 0.0);
    COLORREF color = RGB(0, 255, 255);
    int width = 1;
};

class CImageViewPane : public CWnd
{
public:
    CImageViewPane();

    BOOL CreatePane(CWnd* pParentWnd, UINT controlId, const CString& title);
    void SetImage(const cv::Mat& image);
    void SetOverlay(const std::vector<ImageOverlayPoint>& points, const std::vector<ImageOverlayLine>& lines);
    void ClearOverlay();
    void ClearImage();
    void ResetView();

protected:
    afx_msg void OnPaint();
    afx_msg BOOL OnEraseBkgnd(CDC* pDC);
    afx_msg void OnSize(UINT nType, int cx, int cy);
    afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
    afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
    afx_msg void OnMouseMove(UINT nFlags, CPoint point);
    afx_msg BOOL OnMouseWheel(UINT nFlags, short zDelta, CPoint pt);
    afx_msg void OnDropFiles(HDROP hDropInfo);

    DECLARE_MESSAGE_MAP()

private:
    void DrawContent(CDC& dc, const CRect& clientRect);
    void FitImageToClient();
    double GetMinScaleForClient(const CRect& clientRect) const;
    CPoint ImageToClient(const cv::Point2d& point) const;
    cv::Point2d ClientToImage(const CPoint& point) const;
    cv::Mat NormalizeToBgra(const cv::Mat& image) const;

private:
    CString m_title;
    cv::Mat m_image;
    std::vector<ImageOverlayPoint> m_overlayPoints;
    std::vector<ImageOverlayLine> m_overlayLines;
    double m_scale;
    cv::Point2d m_offset;
    bool m_isDragging;
    bool m_keepImageFitted;
    CPoint m_lastMousePoint;
};
