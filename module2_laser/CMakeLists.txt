# module2_laser: 激光标定（CUDA）
find_package(OpenCV 4.13 REQUIRED COMPONENTS
    core imgproc calib3d cudaimgproc cudaarithm cudafilters flann)
find_package(Eigen3 3.4 REQUIRED)

file(GLOB_RECURSE M2_OP_CPP "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.cpp")
file(GLOB_RECURSE M2_OP_CU  "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.cu")
file(GLOB_RECURSE M2_OP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.h")
file(GLOB_RECURSE M2_TOP_TESTS_CPP "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
set(M2_SOURCES "")
set(M2_TESTS "")
foreach(f ${M2_OP_CPP} ${M2_OP_CU} ${M2_TOP_TESTS_CPP})
    if(f MATCHES "tests[\\\\/]test_[^/\\\\]*\\.cpp$")
        list(APPEND M2_TESTS ${f})
    elseif(f MATCHES "tests[\\\\/]")
        # 跳过 tests 目录内非 test_ 文件
    else()
        list(APPEND M2_SOURCES ${f})
    endif()
endforeach()

set(M2_INC_DIRS "")
foreach(h ${M2_OP_HEADERS})
    get_filename_component(d ${h} DIRECTORY)
    list(APPEND M2_INC_DIRS ${d})
endforeach()
list(REMOVE_DUPLICATES M2_INC_DIRS)

if(M2_SOURCES)
    add_library(fc2_ops STATIC ${M2_SOURCES})
    target_include_directories(fc2_ops PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/operators
        ${M2_INC_DIRS})
    target_link_libraries(fc2_ops PUBLIC
        ${OpenCV_LIBS} Eigen3::Eigen spdlog::spdlog nlohmann_json::nlohmann_json)
    target_compile_definitions(fc2_ops PUBLIC BUILD_CUDA=1)
    target_compile_definitions(fc2_ops PRIVATE LM_ENABLE_TIMING=1)
    if(MSVC)
        target_compile_options(fc2_ops PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
    endif()
endif()

# === fc2_io: calib_io (读 config/handoff/姿态目录, 写 laser_calib.json) ===
# 经验 #3 实测修订: find_package REQUIRED 没列 imgcodecs → ${OpenCV_LIBS} 不带它。
# Release OpenCV 装在 ${OpenCV_DIR} 下有 opencv_imgcodecs4130.lib; Debug 无。
# → 探测 imgcodecs 存在才构建 fc2_io/laser_calib (Debug 自动跳过, 不影响算子自测)
set(FC2_IMGCODECS_LIB "")
if(EXISTS "${OpenCV_DIR}/x64/vc17/lib/opencv_imgcodecs4130.lib")
    set(FC2_IMGCODECS_LIB "${OpenCV_DIR}/x64/vc17/lib/opencv_imgcodecs4130.lib")
elseif(EXISTS "${OpenCV_DIR}/lib/opencv_imgcodecs4130.lib")
    set(FC2_IMGCODECS_LIB "${OpenCV_DIR}/lib/opencv_imgcodecs4130.lib")
endif()

if(TARGET fc2_ops AND FC2_IMGCODECS_LIB)
    # 把 laser_calib_cli.cpp 也编进 fc2_io（FC2_BUILDING_LIB 让它跳过 main()）
    add_library(fc2_io STATIC calib_io.cpp calib_io.h
                            calib_runner.cpp calib_runner.h
                            laser_calib_cli.cpp
                            laser_calib_runner_internal.h)
    target_include_directories(fc2_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(fc2_io PRIVATE FC2_BUILDING_LIB)
    target_link_libraries(fc2_io PUBLIC
        fc2_ops ${OpenCV_LIBS} ${FC2_IMGCODECS_LIB}
        nlohmann_json::nlohmann_json spdlog::spdlog)
    if(MSVC)
        target_compile_options(fc2_io PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    endif()

    add_executable(laser_calib laser_calib_cli.cpp)
    target_link_libraries(laser_calib PRIVATE fc2_ops fc2_io)
    if(MSVC)
        target_compile_options(laser_calib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    endif()
    fc_deploy_crt(laser_calib)
    fc_deploy_opencv_dlls(laser_calib)
    fc_setup_vs_debugger(laser_calib "${CMAKE_SOURCE_DIR}")
    message(STATUS "fc2_io + laser_calib: ENABLED (imgcodecs at ${FC2_IMGCODECS_LIB})")
else()
    message(WARNING
        "fc2_io + laser_calib: SKIPPED (imgcodecs not found; "
        "Debug OpenCPV limitation per 经验 #3). "
        "Algorithm self-tests (fc2_ops + tests) remain available.")
endif()

if(FC_BUILD_TESTS)
    foreach(t ${M2_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        set(tname "fc2_${tname}")  # 加 fc2_ 前缀避免与模块1 测试同名冲突
        add_executable(${tname} ${t})
        if(TARGET fc2_ops)
            target_link_libraries(${tname} PRIVATE fc2_ops GTest::gtest_main)
            target_include_directories(${tname} PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/operators ${M2_INC_DIRS}
                ${CMAKE_CURRENT_SOURCE_DIR})
            target_compile_definitions(${tname} PRIVATE WITH_CUDA_TESTS=1)
            # 顶层 tests/test_*.cpp 需要 fc2_io（含 imgcodecs）和 laser_calib target
            if(TARGET fc2_io)
                target_link_libraries(${tname} PRIVATE fc2_io)
            endif()
            if(MSVC)
                target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
            endif()
            add_test(NAME ${tname} COMMAND ${tname})
            # 冒烟/e2e 测试需要 laser_calib.exe 路径
            if(TARGET laser_calib AND tname MATCHES "laser_calib")
                set_tests_properties(${tname} PROPERTIES
                    ENVIRONMENT "LASER_CALIB_EXE=$<TARGET_FILE:laser_calib>")
            endif()
            fc_deploy_crt(${tname})
        endif()
    endforeach()
endif()
