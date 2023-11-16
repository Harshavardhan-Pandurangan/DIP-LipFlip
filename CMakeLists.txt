cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(lip_flip)

### ---[ Find universal dependencies
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})

#find_package(PkgConfig REQUIRED)
find_package( OpenCV )

include_directories(${OpenCV_INCLUDE_DIRS})

set(LIPFLIP_SRC main.cpp
		DetectionTools.cpp
		LaplacianBlending.cpp
		ImageTools.cpp)
set(LIPFLIP_HDR
		include/DetectionTools.hpp
		include/LaplacianBlending.hpp
		include/ImageTools.hpp)

add_executable(lip_flip ${LIPFLIP_SRC} ${LIPFLIP_HDR})
target_link_libraries(lip_flip ${OpenCV_LIBS})

