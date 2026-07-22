# gui_qt: 统一 GUI（采集 / 相机标定 / 激光标定）
#
# 直接链接 fc1_io（含 calib_runner）和 fc2_io（含 calib_runner），不再走 QProcess。
# 相机走 ICameraSource 抽象：simulated 始终可用；hikvision 在 CMake 找到 MVS 时启用。

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)

# 探测 Hikvision MVS SDK（可选，找到才编译 HikvisionCameraSource.cpp）
set(MVS_INCLUDE_DIR "C:/Program Files (x86)/MVS/Development/Includes")
set(MVS_LIB_DIR     "C:/Program Files (x86)/MVS/Development/Libraries/win64")
if(EXISTS "${MVS_INCLUDE_DIR}/MvCameraControl.h"
   AND EXISTS "${MVS_LIB_DIR}/MvCameraControl.lib")
    set(FC_HAVE_MVS ON)
    message(STATUS "MVS SDK: found (Hikvision)")
else()
    set(FC_HAVE_MVS OFF)
    message(STATUS "MVS SDK: not found (camera = simulated only)")
endif()

# 探测 Daheng Galaxy SDK（优先 Daheng 官方安装路径，其次 LeadScanK2 项目内嵌）
set(GALAXY_SDK_ROOT "${GALAXY_SDK_ROOT}" CACHE PATH
    "Daheng Galaxy SDK 根（自动探测；含 GalaxyIncludes.h 与 GxIAPICPPEx.lib）")

# 候选根（按官方安装布局：根下有 Development/C++ SDK/inc、APIDll/Win64、GenICam/bin/Win64_x64）
set(_galaxy_official_dirs
    "C:/Program Files/Daheng Imaging/GalaxySDK"
    "C:/Program Files (x86)/Daheng Imaging/GalaxySDK")
# 候选根（项目内嵌：根下直接是 inc/lib）
set(_galaxy_embedded_dirs
    "E:/workfold/20260509intergrate/LeadScanK2/SDK")

# 探测函数：找到根目录；设置 GALAXY_INC_DIR / GALAXY_LIB_DIR / GALAXY_RUNTIME_DIRS
macro(fc_find_galaxy out_root out_inc out_lib out_runtime)
    set(${out_root} "")
    set(${out_inc} "")
    set(${out_lib} "")
    set(${out_runtime} "")
    # 官方安装布局
    foreach(_c ${_galaxy_official_dirs})
        if(EXISTS "${_c}/Development/C++ SDK/inc/GalaxyIncludes.h"
           AND EXISTS "${_c}/Development/C++ SDK/lib/x64/GxIAPICPPEx.lib")
            set(${out_root}     "${_c}")
            set(${out_inc}      "${_c}/Development/C++ SDK/inc")
            set(${out_lib}      "${_c}/Development/C++ SDK/lib/x64")
            set(${out_runtime}  "${_c}/APIDll/Win64;${_c}/GenICam/bin/Win64_x64")
            break()
        endif()
    endforeach()
    # 项目内嵌布局
    if(NOT ${out_root})
        foreach(_c ${_galaxy_embedded_dirs})
            if(EXISTS "${_c}/inc/GalaxyIncludes.h"
               AND EXISTS "${_c}/lib/x64/GxIAPICPPEx.lib")
                set(${out_root}    "${_c}")
                set(${out_inc}     "${_c}/inc")
                set(${out_lib}     "${_c}/lib/x64")
                # 内嵌版不带 runtime DLL；fallback 在宏外处理
                set(${out_runtime} "")
                break()
            endif()
        endforeach()
    endif()
endmacro()

if(NOT GALAXY_SDK_ROOT)
    fc_find_galaxy(GALAXY_SDK_ROOT GALAXY_INC_DIR GALAXY_LIB_DIR GALAXY_RUNTIME_DIRS)
endif()

if(EXISTS "${GALAXY_INC_DIR}/GalaxyIncludes.h" AND EXISTS "${GALAXY_LIB_DIR}/GxIAPICPPEx.lib")
    set(FC_HAVE_GALAXY ON)
    message(STATUS "Galaxy SDK: ${GALAXY_SDK_ROOT}")
    message(STATUS "  inc:      ${GALAXY_INC_DIR}")
    message(STATUS "  lib:      ${GALAXY_LIB_DIR}")
    message(STATUS "  runtime:  ${GALAXY_RUNTIME_DIRS}")
endif()

set(GUI_SOURCES
    main.cpp
    MainWindow.cpp MainWindow.h
    AcquisitionTab.cpp AcquisitionTab.h
    CameraCalibTab.cpp CameraCalibTab.h
    LaserCalibTab.cpp LaserCalibTab.h
    CalibWorker.cpp CalibWorker.h
    SpdlogBridge.cpp SpdlogBridge.h
    PreviewWidget.cpp PreviewWidget.h
    ResultView.cpp ResultView.h
    gui_common.cpp gui_common.h
    camera/ICameraSource.cpp camera/ICameraSource.h
    camera/SimulatedCameraSource.cpp camera/SimulatedCameraSource.h
    camera/HikvisionCameraSource.cpp camera/HikvisionCameraSource.h
    camera/GalaxyCameraSource.cpp camera/GalaxyCameraSource.h
    camera/StereoCameraRig.cpp camera/StereoCameraRig.h
    scanner/ScannerControl.cpp scanner/ScannerControl.h
)

add_executable(factory_calib_gui WIN32 ${GUI_SOURCES})

target_include_directories(factory_calib_gui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/camera
    # 模块算子头（直接调用 fc::runCameraCalib / extractCorners 等）
    ${CMAKE_SOURCE_DIR}/module1_camera
    ${CMAKE_SOURCE_DIR}/module1_camera/operators
    ${CMAKE_SOURCE_DIR}/module2_laser
    ${CMAKE_SOURCE_DIR}/module2_laser/operators)

# 链接业务库（必须是相同 build）
target_link_libraries(factory_calib_gui PRIVATE
    fc1_io
    fc1_chess
    fc1_ops
    fc2_ops
    ${OpenCV_LIBS}
    Qt5::Widgets
    Qt5::Concurrent
    Qt5::SerialPort
    nlohmann_json::nlohmann_json
    spdlog::spdlog)

if(TARGET fc2_io)
    target_link_libraries(factory_calib_gui PRIVATE fc2_io)
    target_compile_definitions(factory_calib_gui PRIVATE FC_HAVE_LASER=1)
endif()

if(FC_HAVE_MVS)
    target_compile_definitions(factory_calib_gui PRIVATE FC_HAVE_MVS=1)
    target_include_directories(factory_calib_gui PRIVATE "${MVS_INCLUDE_DIR}")
    target_link_libraries(factory_calib_gui PRIVATE
        "${MVS_LIB_DIR}/MvCameraControl.lib")
endif()

if(FC_HAVE_GALAXY)
    target_compile_definitions(factory_calib_gui PRIVATE FC_HAVE_GALAXY=1)
    target_include_directories(factory_calib_gui PRIVATE "${GALAXY_INC_DIR}")
    target_link_libraries(factory_calib_gui PRIVATE
        "${GALAXY_LIB_DIR}/GxIAPICPPEx.lib")

    # 拷 Galaxy 运行 DLL（只拷 Galaxy 自家的，跳过 msvcp/vcruntime 避免和系统 DLL 冲突）
    file(GLOB _galaxy_api_dlls    "${GALAXY_SDK_ROOT}/APIDll/Win64/Gx*.dll"
                                  "${GALAXY_SDK_ROOT}/APIDll/Win64/DxImageProc.dll")
    file(GLOB _galaxy_genicam_dlls "${GALAXY_SDK_ROOT}/GenICam/bin/Win64_x64/GCBase*.dll"
                                   "${GALAXY_SDK_ROOT}/GenICam/bin/Win64_x64/Gen*.dll"
                                   "${GALAXY_SDK_ROOT}/GenICam/bin/Win64_x64/log4cpp*.dll"
                                   "${GALAXY_SDK_ROOT}/GenICam/bin/Win64_x64/XmlParser*.dll"
                                   "${GALAXY_SDK_ROOT}/GenICam/bin/Win64_x64/Log_*.dll"
                                   "${GALAXY_SDK_ROOT}/GenICam/bin/Win64_x64/MathParser*.dll"
                                   "${GALAXY_SDK_ROOT}/GenICam/bin/Win64_x64/NodeMapData*.dll")
    foreach(_dll ${_galaxy_api_dlls} ${_galaxy_genicam_dlls})
        add_custom_command(TARGET factory_calib_gui POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}" "$<TARGET_FILE_DIR:factory_calib_gui>"
            VERBATIM)
    endforeach()
endif()

if(MSVC)
    target_compile_options(factory_calib_gui PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /bigobj>)
endif()

# 拷贝 Qt5 DLL 到 exe 同目录（WINDEPLOYQT）
if(WIN32)
    find_program(QT_WINDEPLOYQT_EXE
        NAMES windeployqt
        HINTS "${Qt5_DIR}/../../../bin"
              "$ENV{QTDIR}/bin"
        NO_DEFAULT_PATH)
    if(NOT QT_WINDEPLOYQT_EXE)
        find_program(QT_WINDEPLOYQT_EXE NAMES windeployqt)
    endif()
    if(QT_WINDEPLOYQT_EXE)
        add_custom_command(TARGET factory_calib_gui POST_BUILD
            COMMAND ${QT_WINDEPLOYQT_EXE}
                --no-translations --no-system-d3d-compiler --no-opengl-sw
                "$<TARGET_FILE:factory_calib_gui>"
            COMMENT "部署 Qt5 DLL 到 factory_calib_gui")
        message(STATUS "windeployqt: ${QT_WINDEPLOYQT_EXE}")
    else()
        message(WARNING "windeployqt 未找到")
    endif()
endif()

fc_deploy_crt(factory_calib_gui)
fc_deploy_opencv_dlls(factory_calib_gui)
fc_setup_vs_debugger(factory_calib_gui "${CMAKE_SOURCE_DIR}")
