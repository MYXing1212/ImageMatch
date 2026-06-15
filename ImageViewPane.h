#pragma once

#include "framework.h"

constexpr UINT WM_IMAGE_PANE_FILE_DROPPED = WM_APP + 101;

struct ImagePaneDropPayload
{
    UINT paneId = 0;
    CString filePath;
};

class CImageViewPane : public CWnd
{
public:
    CImageViewPane();

    BOOL CreatePane(CWnd* pParentWnd, UINT controlId, const CString& title);
    void SetImage(const cv::Mat& image);
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
    cv::Point2d ClientToImage(const CPoint& point) const;
    cv::Mat NormalizeToBgr(const cv::Mat& image) const;

private:
    CString m_title;
    cv::Mat m_image;
    double m_scale;
    cv::Point2d m_offset;
    bool m_isDragging;
    CPoint m_lastMousePoint;
};
