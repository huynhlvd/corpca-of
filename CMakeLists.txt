# the minimum version of CMake required
    cmake_minimum_required(VERSION 2.8)
project( CORPCA-OF)
find_package( OpenCV REQUIRED)# video cudaarithm cudaoptflow highgui)
find_package( CUDA QUIET COMPONENTS)

    # required modules for our task
    include(CheckIncludeFile)
    include(CheckIncludeFileCXX)
    include(CheckIncludeFiles)
    include(ExternalProject)

    # we add the files in the include directory to be included
    # while compiling all the source code
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
	include_directories( ${OpenCV_INCLUDE_DIRS} )
	include_directories( ${CUDA_INCLUDE_DIRS} QUIET)

    # we add the sub-directories that we want CMake to scan
    add_subdirectory(include)
    add_subdirectory(src)

add_library( corpcaOF SHARED src/corpcaOF.cpp src/VideoSource.cpp src/VideoSourceManager.cpp src/corpcaOFUtils.cpp src/priorInitialization.cpp src/inexact_alm_rpca.cpp src/ramsia.cpp)
add_executable( dempCORPCA-OF src/demoCorpcaOF.cpp  )
target_link_libraries( corpcaOF ${OpenCV_LIBS} ${CUDA_LIBS} )
target_link_libraries (demoCORPCA-OF LINK_PUBLIC corpcaOF)
#install(TARGETS test_CMake DESTINATION bin)
#add_library(${OpenCV_LIBS}/bin)

