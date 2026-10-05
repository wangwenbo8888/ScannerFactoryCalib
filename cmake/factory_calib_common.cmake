# 复刻自主工程根 CMakeLists.txt:91-106
# 自动探测 VS2022（任意 edition）+ 最新 MSVC redist CRT 目录
if(NOT FC_MSVC_REDIST_CRT_DIR)
    file(GLOB _fc_vs_editions
        "C:/Program Files/Microsoft Visual Studio/2022/Community"
        "C:/Program Files/Microsoft Visual Studio/2022/Professional"
        "C:/Program Files/Microsoft Visual Studio/2022/Enterprise"
        "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools")
    foreach(_ed ${_fc_vs_editions})
        file(GLOB _fc_redist_ver RELATIVE "${_ed}/VC/Redist/MSVC"
            "${_ed}/VC/Redist/MSVC/14.*")
        list(SORT _fc_redist_ver COMPARE STRING ORDER DESCENDING)
        foreach(_v ${_fc_redist_ver})
            if(EXISTS "${_ed}/VC/Redist/MSVC/${_v}/x64/Microsoft.VC143.CRT/msvcp140.dll")
                set(FC_MSVC_REDIST_CRT_DIR "${_ed}/VC/Redist/MSVC/${_v}/x64/Microsoft.VC143.CRT")
                break()
            endif()
        endforeach()
        if(FC_MSVC_REDIST_CRT_DIR)
            break()
        endif()
    endforeach()
    set(FC_MSVC_REDIST_CRT_DIR "${FC_MSVC_REDIST_CRT_DIR}" CACHE PATH
        "VS2022 MSVC redist CRT DLL dir (含 msvcp140/vcruntime140)" FORCE)
    unset(_fc_vs_editions)
    unset(_fc_redist_ver)
    unset(_v)
    unset(_ed)
endif()

# E(2026-09-03, 自源算法线移植 2026-10-05): OpenMP 运行时 vcomp140.dll best-effort 探测
# （redist 各版本取最大；找不到仅告警不阻断——模块2 curve_map 行级并行链接 OpenMP 需要它）
get_filename_component(FC_MSVC_REDIST_MSVC_DIR "${FC_MSVC_REDIST_CRT_DIR}" DIRECTORY)
get_filename_component(FC_MSVC_REDIST_MSVC_DIR "${FC_MSVC_REDIST_MSVC_DIR}" DIRECTORY)
get_filename_component(FC_MSVC_REDIST_MSVC_DIR "${FC_MSVC_REDIST_MSVC_DIR}" DIRECTORY)
file(GLOB FC_MSVC_REDIST_OMP_GLOB
    "${FC_MSVC_REDIST_MSVC_DIR}/*/x64/Microsoft.VC143.OpenMP/vcomp140.dll")
set(FC_MSVC_REDIST_OMP_DLL "")
if(FC_MSVC_REDIST_OMP_GLOB)
    list(SORT FC_MSVC_REDIST_OMP_GLOB)
    list(REVERSE FC_MSVC_REDIST_OMP_GLOB)
    list(GET FC_MSVC_REDIST_OMP_GLOB 0 FC_MSVC_REDIST_OMP_DLL)
else()
    message(WARNING "fc_deploy_crt: 未找到 vcomp140.dll (OpenMP 运行时, "
        "${FC_MSVC_REDIST_MSVC_DIR}/*/x64/Microsoft.VC143.OpenMP/ 下无匹配) —— 跳过其部署")
endif()

function(fc_deploy_crt target)
    if(NOT FC_MSVC_REDIST_CRT_DIR OR NOT EXISTS "${FC_MSVC_REDIST_CRT_DIR}/msvcp140.dll")
        message(STATUS "fc_deploy_crt: 跳过（CRT 目录无效）")
        return()
    endif()
    set(FC_DEPLOY_DLLS
        "${FC_MSVC_REDIST_CRT_DIR}/msvcp140.dll"
        "${FC_MSVC_REDIST_CRT_DIR}/vcruntime140.dll"
        "${FC_MSVC_REDIST_CRT_DIR}/vcruntime140_1.dll")
    if(FC_MSVC_REDIST_OMP_DLL AND EXISTS "${FC_MSVC_REDIST_OMP_DLL}")
        list(APPEND FC_DEPLOY_DLLS "${FC_MSVC_REDIST_OMP_DLL}")
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${FC_DEPLOY_DLLS}
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "部署 VS2022 CRT DLL 到 ${target}")
endfunction()

# -----------------------------------------------------------------------------
# fc_deploy_opencv_dlls(target)
# 把 OpenCV Release DLL（opencv_world4130.dll 等）拷到 target 的 exe 同目录，
# 让 VS F5 调试 / 双击运行 都能找到，无需配置全局 PATH。
# -----------------------------------------------------------------------------
function(fc_deploy_opencv_dlls target)
    set(_opencv_bin "${OpenCV_DIR}/x64/vc17/bin")
    if(NOT EXISTS "${_opencv_bin}/opencv_world4130.dll")
        # 可能是分立 lib（非 world）；glob 一遍
        file(GLOB _opencv_dlls_test "${_opencv_bin}/opencv_*4130.dll")
        if(NOT _opencv_dlls_test)
            message(STATUS "fc_deploy_opencv_dlls: 跳过（${_opencv_bin}/opencv_*4130.dll 不存在）")
            return()
        endif()
    endif()
    file(GLOB _opencv_dlls
        "${_opencv_bin}/opencv_*4130.dll")
    foreach(_dll ${_opencv_dlls})
        if(EXISTS "${_dll}")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_dll}" "$<TARGET_FILE_DIR:${target}>"
                VERBATIM)
        endif()
    endforeach()
    message(STATUS "fc_deploy_opencv_dlls: 部署 ${_opencv_bin} → ${target} (${ARGC} dll)")
endfunction()

# -----------------------------------------------------------------------------
# fc_setup_vs_debugger(target working_dir)
# 配置 VS2022 调试器：工作目录 + PATH 注入 OpenCV/Qt bin，让 F5/Ctrl+F5 可用。
# -----------------------------------------------------------------------------
function(fc_setup_vs_debugger target working_dir)
    set_target_properties(${target} PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${working_dir}"
        VS_DEBUGGER_ENVIRONMENT "PATH=${OpenCV_DIR}/x64/vc17/bin;%PATH%"
    )
endfunction()
