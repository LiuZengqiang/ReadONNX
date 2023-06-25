cmake_minimum_required(VERSION 3.0.0)
project(ReadONNX VERSION 0.1.0 LANGUAGES C CXX)

find_package(Protobuf REQUIRED)

include_directories(include)

add_executable(ReadONNX main.cpp ./src/onnx.pd.cc)

target_link_libraries(ReadONNX
    PUBLIC
    ${PROTOBUF_LIBRARIES}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)