@echo off
rem ============================================================
rem  factory_calib GUI 启动器
rem  自动设置 OpenCV / Qt5 DLL 路径并启动 GUI
rem ============================================================
setlocal

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

rem 探测构建产物（优先 build_fc_all，其次 build_fc）
set "GUI_DIR="
if exist "%ROOT%\build_fc_all\gui_qt\Release\factory_calib_gui.exe" set "GUI_DIR=%ROOT%\build_fc_all\gui_qt\Release"
if not defined GUI_DIR if exist "%ROOT%\build_fc\gui_qt\Release\factory_calib_gui.exe" set "GUI_DIR=%ROOT%\build_fc\gui_qt\Release"

if not defined GUI_DIR (
    echo [error] GUI exe 未找到。请先执行构建。
    exit /b 1
)

set "PATH=%GUI_DIR%;F:\opencv4.13\install\x64\vc17\bin;D:\QTInstall\5.9.3\5.9.3\msvc2015_64\bin;%PATH%"

rem cwd = 工程根（让 data_in/data_out 相对路径有意义）
cd /d "%ROOT%"

start "" "%GUI_DIR%\factory_calib_gui.exe" --root "%ROOT%"
endlocal
