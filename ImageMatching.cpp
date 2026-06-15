#include "framework.h"
#include "ImageMatching.h"
#include "MainDialog.h"

CImageMatchingApp theApp;

BOOL CImageMatchingApp::InitInstance()
{
    CWinApp::InitInstance();

    CMainDialog dialog;
    m_pMainWnd = &dialog;
    dialog.DoModal();
    return FALSE;
}
