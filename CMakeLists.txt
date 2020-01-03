cmake_minimum_required(VERSION 3.6)

project(beblid_demo)

set(CMAKE_CXX_STANDARD 11)

# If OpenCV is not installed in the system libraries set the variable OpenCV_DIR
set(OpenCV_DIR /home/iago/graffter/libs/opencv/opencv-4.1.0/linux_contrib_debug/install/lib/cmake/opencv4)
# Find the OpenCV library
find_package(OpenCV)

# Include the headers of OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Create an executable with the BEBLID demo to describe and match two images
add_executable(beblid_demo demo.cpp BEBLID.cpp)

# Link the demo executable with OpenCV
target_link_libraries(beblid_demo ${OpenCV_LIBS})
