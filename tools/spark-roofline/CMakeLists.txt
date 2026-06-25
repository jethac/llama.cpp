set(TARGET llama-spark-roofline)
set(POSTMMA_TARGET llama-spark-postmma)
set(REALHANDOFF_TARGET llama-spark-realhandoff)
set(FRAGQUEUE_TARGET llama-spark-fragqueue)
set(REGQUEUE_TARGET llama-spark-regqueue)
set(LATEEXTRACT_TARGET llama-spark-lateextract)
set(ROLLINGLATE_TARGET llama-spark-rollinglate)
set(GROUPSTREAM_TARGET llama-spark-groupstream)
set(KQSTAGE_TARGET llama-spark-kqstage)
set(KQ256_TARGET llama-spark-kq256)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

add_executable(${TARGET} spark-roofline.cu host-runtime.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${TARGET} PRIVATE CUDA::cudart)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

add_executable(${POSTMMA_TARGET} spark-postmma.cu host-runtime.cpp)
target_link_libraries(${POSTMMA_TARGET} PRIVATE CUDA::cudart)
target_compile_features(${POSTMMA_TARGET} PRIVATE cxx_std_17)

add_executable(${REALHANDOFF_TARGET} spark-realhandoff.cu host-runtime.cpp)
target_include_directories(${REALHANDOFF_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${REALHANDOFF_TARGET} PRIVATE CUDA::cudart)
target_compile_features(${REALHANDOFF_TARGET} PRIVATE cxx_std_17)

add_executable(${FRAGQUEUE_TARGET} spark-fragqueue.cu host-runtime.cpp)
target_include_directories(${FRAGQUEUE_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${FRAGQUEUE_TARGET} PRIVATE CUDA::cudart)
target_compile_features(${FRAGQUEUE_TARGET} PRIVATE cxx_std_17)

add_executable(${REGQUEUE_TARGET} spark-regqueue.cu host-runtime.cpp)
target_include_directories(${REGQUEUE_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${REGQUEUE_TARGET} PRIVATE CUDA::cudart)
target_compile_features(${REGQUEUE_TARGET} PRIVATE cxx_std_17)

add_executable(${LATEEXTRACT_TARGET} spark-lateextract.cu host-runtime.cpp)
target_include_directories(${LATEEXTRACT_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${LATEEXTRACT_TARGET} PRIVATE CUDA::cudart)
target_compile_features(${LATEEXTRACT_TARGET} PRIVATE cxx_std_17)

add_executable(${ROLLINGLATE_TARGET} spark-rollinglate.cu host-runtime.cpp)
target_include_directories(${ROLLINGLATE_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${ROLLINGLATE_TARGET} PRIVATE CUDA::cudart)
target_compile_features(${ROLLINGLATE_TARGET} PRIVATE cxx_std_17)

add_executable(${GROUPSTREAM_TARGET} spark-groupstream.cu host-runtime.cpp)
target_include_directories(${GROUPSTREAM_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${GROUPSTREAM_TARGET} PRIVATE CUDA::cudart)
target_compile_features(${GROUPSTREAM_TARGET} PRIVATE cxx_std_17)

add_executable(${KQSTAGE_TARGET} spark-kqstage.cu host-runtime.cpp)
target_include_directories(${KQSTAGE_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${KQSTAGE_TARGET} PRIVATE CUDA::cudart)
target_compile_features(${KQSTAGE_TARGET} PRIVATE cxx_std_17)

add_executable(${KQ256_TARGET} spark-kq256.cu host-runtime.cpp)
target_include_directories(${KQ256_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${KQ256_TARGET} PRIVATE CUDA::cudart)
target_compile_features(${KQ256_TARGET} PRIVATE cxx_std_17)

set(SPARK_ROOFLINE_CUDA_FLAGS)
if (MSVC)
    list(APPEND SPARK_ROOFLINE_CUDA_FLAGS -Xcompiler=/utf-8)
endif()
foreach(ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX REPLACE "(-real|-virtual)$" "" ARCH_BASE "${ARCH}")
    if (ARCH MATCHES "-virtual$")
        set(ARCH_CODE "compute_${ARCH_BASE}")
    elseif(ARCH MATCHES "-real$")
        set(ARCH_CODE "sm_${ARCH_BASE}")
    else()
        set(ARCH_CODE "[compute_${ARCH_BASE},sm_${ARCH_BASE}]")
    endif()
    list(APPEND SPARK_ROOFLINE_CUDA_FLAGS "--generate-code=arch=compute_${ARCH_BASE},code=${ARCH_CODE}")
endforeach()
target_compile_options(${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_ROOFLINE_CUDA_FLAGS}>")
target_compile_options(${POSTMMA_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_ROOFLINE_CUDA_FLAGS}>")
target_compile_options(${REALHANDOFF_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_ROOFLINE_CUDA_FLAGS}>")
target_compile_options(${FRAGQUEUE_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_ROOFLINE_CUDA_FLAGS}>")
target_compile_options(${REGQUEUE_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_ROOFLINE_CUDA_FLAGS}>")
target_compile_options(${LATEEXTRACT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_ROOFLINE_CUDA_FLAGS}>")
target_compile_options(${ROLLINGLATE_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_ROOFLINE_CUDA_FLAGS}>")
target_compile_options(${GROUPSTREAM_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_ROOFLINE_CUDA_FLAGS}>")
target_compile_options(${KQSTAGE_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_ROOFLINE_CUDA_FLAGS}>")
target_compile_options(${KQ256_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SPARK_ROOFLINE_CUDA_FLAGS}>")

set_target_properties(${TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set_target_properties(${POSTMMA_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set_target_properties(${REALHANDOFF_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set_target_properties(${FRAGQUEUE_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set_target_properties(${REGQUEUE_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set_target_properties(${LATEEXTRACT_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set_target_properties(${ROLLINGLATE_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set_target_properties(${GROUPSTREAM_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set_target_properties(${KQSTAGE_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set_target_properties(${KQ256_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} ${POSTMMA_TARGET} ${REALHANDOFF_TARGET} ${FRAGQUEUE_TARGET} ${REGQUEUE_TARGET} ${LATEEXTRACT_TARGET} ${ROLLINGLATE_TARGET} ${GROUPSTREAM_TARGET} ${KQSTAGE_TARGET} ${KQ256_TARGET} RUNTIME)
endif()
