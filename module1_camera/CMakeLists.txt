# module1_camera: 相机标定（纯 CPU）
# 复刻 modules/09_operatorlib/CMakeLists.txt 的 glob + include 策略，但只 CPU

find_package(OpenCV 4.13 REQUIRED COMPONENTS core imgproc imgcodecs calib3d flann)
find_package(Eigen3 3.4 REQUIRED)

# === 收集算子源 ===
file(GLOB_RECURSE M1_OP_CPP "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.cpp")
file(GLOB_RECURSE M1_TESTS_CPP "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
file(GLOB_RECURSE M1_OP_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/operators/*.h")
set(M1_SOURCES "")
set(M1_TESTS "")
foreach(f ${M1_OP_CPP} ${M1_TESTS_CPP})
    if(f MATCHES "tests[\\\\/]test_[^/\\\\]*\\.cpp$")
        list(APPEND M1_TESTS ${f})
    elseif(f MATCHES "tests[\\\\/]")
        # 跳过 tests 目录内非 test_ 文件
    else()
        list(APPEND M1_SOURCES ${f})
    endif()
endforeach()

# 每个算子目录都加入 include 路径（跨算子 include）
set(M1_INC_DIRS "")
foreach(h ${M1_OP_HEADERS})
    get_filename_component(d ${h} DIRECTORY)
    list(APPEND M1_INC_DIRS ${d})
endforeach()
list(REMOVE_DUPLICATES M1_INC_DIRS)

# === 算子静态库 ===
if(M1_SOURCES)
    add_library(fc1_ops STATIC ${M1_SOURCES})
    target_include_directories(fc1_ops PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/operators   # "common/xxx.h"
        ${M1_INC_DIRS})                          # 跨算子 include
    target_link_libraries(fc1_ops PUBLIC
        ${OpenCV_LIBS} Eigen3::Eigen spdlog::spdlog nlohmann_json::nlohmann_json)
    if(MSVC)
        target_compile_options(fc1_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    endif()
endif()

# === 模块1 自有非算子源（chessboard_corner） ===
add_library(fc1_chess STATIC chessboard_corner.cpp chessboard_corner.h)
target_include_directories(fc1_chess PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fc1_chess PUBLIC ${OpenCV_LIBS} spdlog::spdlog)

# === 模块1 I/O（calib_io 读 config/温度/图像 + 写交接 json） ===
add_library(fc1_io STATIC calib_io.cpp calib_io.h calib_runner.cpp calib_runner.h)
target_include_directories(fc1_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fc1_io PUBLIC fc1_ops ${OpenCV_LIBS} nlohmann_json::nlohmann_json spdlog::spdlog)

# === camera_calib.exe ===
add_executable(camera_calib camera_calib_cli.cpp)
target_link_libraries(camera_calib PRIVATE fc1_ops fc1_chess fc1_io ${OpenCV_LIBS} nlohmann_json::nlohmann_json spdlog::spdlog)
fc_deploy_crt(camera_calib)
fc_deploy_opencv_dlls(camera_calib)
fc_setup_vs_debugger(camera_calib "${CMAKE_SOURCE_DIR}")

# === 测试 ===
if(FC_BUILD_TESTS)
    foreach(t ${M1_TESTS})
        get_filename_component(tname ${t} NAME_WE)
        add_executable(${tname} ${t})
        target_link_libraries(${tname} PRIVATE fc1_ops fc1_chess GTest::gtest_main)
        target_include_directories(${tname} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}                         # chessboard_corner.h
            ${CMAKE_CURRENT_SOURCE_DIR}/operators ${M1_INC_DIRS})
        target_compile_definitions(${tname} PRIVATE WITH_CUDA_TESTS=0)
        if(MSVC)
            target_compile_options(${tname} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
        endif()
        add_test(NAME ${tname} COMMAND ${tname})
        fc_deploy_crt(${tname})
    endforeach()
    # 模块1 自有测试（Phase 2/3 添加，glob 自带捕获）
endif()
