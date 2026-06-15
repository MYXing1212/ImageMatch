#pragma once

#include "framework.h"
#include "ImageViewPane.h"
#include "RegistrationEngine.h"
#include "resource.h"

class CMainDialog : public CDialogEx
{
public:
    CMainDialog();

    enum
    {
        IDD = IDD_IMAGE_MATCHING_DIALOG
    };

protected:
    virtual void DoDataExchange(CDataExchange* pDX);
    virtual BOOL OnInitDialog();

    afx_msg void OnBnClickedLoadSrc();
    afx_msg void OnBnClickedLoadTarget();
    afx_msg void OnBnClickedMethodRange(UINT controlId);
    afx_msg void OnBnClickedReset();
    afx_msg void OnSize(UINT nType, int cx, int cy);
    afx_msg LRESULT OnImagePaneFileDropped(WPARAM wParam, LPARAM lParam);

    DECLARE_MESSAGE_MAP()

private:
    bool LoadImage(bool isSourceImage);
    bool LoadImageFromPath(const CString& filePath, bool isSourceImage);
    void RunRegistration(RegistrationMethod method);
    void UpdateResultText(const CString& text);
    void RepositionLayout();
    CString FormatRegistrationSummary() const;
    CString BuildResultSummary(const RegistrationResult& result) const;
    RegistrationMethod ResolveMethodFromButton(UINT controlId) const;

private:
    CEdit m_resultEdit;
    CImageViewPane m_srcPane;
    CImageViewPane m_targetPane;
    CImageViewPane m_resultPane;

    cv::Mat m_srcImage;
    cv::Mat m_targetImage;
    cv::Mat m_stitchedImage;
    CString m_resultText;
};
