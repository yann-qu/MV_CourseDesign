cmake_minimum_required(VERSION 3.0.0)

project(MV_CourseDesign)

# Requires OpenCV
set(OpenCV_DIR C:/Users/yann/scoop/apps/opencv/current)

find_package(OpenCV 4 REQUIRED)
message("OpenCV version : ${OpenCV_VERSION}")

add_executable(tutor_dft ${PROJECT_SOURCE_DIR}/src/tutor_dft.cpp)
add_executable(${PROJECT_NAME}
        ${CMAKE_SOURCE_DIR}/src/main.cpp
        ${CMAKE_SOURCE_DIR}/src/MV_CourseDesign.cpp
        ${CMAKE_SOURCE_DIR}/src/StripeMeasure.cpp)

target_link_libraries(tutor_dft PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include/)

target_compile_definitions(${PROJECT_NAME} PUBLIC DEBUG)



