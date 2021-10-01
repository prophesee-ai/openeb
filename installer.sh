#!/bin/sh

RED='\033[0;31m'
GREEN='\033[0;32m'
WHITE='\033[0m'

pybind_version="2.4.3"
openebRoot=$(git rev-parse --show-toplevel)

echo $GREEN"Installing cmake, libusb, eigen3, boost, opencv, glfw3, glew, gtest, python, pybind, numpy."$WHITE
apt-get update
apt-get -y install apt-utils build-essential software-properties-common wget unzip curl
apt-get -y install cmake libopencv-dev git # CMake, OpenCV, and Git
apt-get -y install libusb-1.0-0-dev libeigen3-dev libboost-dev libboost-program-options-dev libboost-filesystem-dev libboost-timer-dev \
libboost-chrono-dev libboost-thread-dev # Boost, Libusb and eigen3
apt-get -y install libglew-dev libglfw3-dev #openGL
apt-get -y install libgtest-dev  # gtest
apt-get -y install python3-pip python3-distutils  # the python package installer
python3 -m pip install pip --upgrade  # upgrade pip
python3 -m pip install numpy #install numpy

#Install pybind
cd $openebRoot/..
wget https://github.com/pybind/pybind11/archive/v$pybind_version.zip
unzip v$pybind_version.zip
cd pybind11-$pybind_version/
mkdir build && cd build
cmake .. -DPYBIND11_TEST=OFF
cmake --build .
make
make install

#Build openEB
cd $openebRoot
echo $GREEN"Generating the openEB build."$WHITE
mkdir build ; cd build
cmake .. -DBUILD_TESTING=OFF
echo $GREEN"Building openEB."$WHITE
cmake --build . --config Release --parallel 4