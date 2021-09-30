#!/bin/sh

RED='\033[0;31m'
GREEN='\033[0;32m'
WHITE='\033[0m'

pybind_version="2.6.0"
openebRoot=$(git rev-parse --show-toplevel)

if [ "$#" -lt 1 ]; then
    PYBIND_HOME="../pybind"
else
    PYBIND_HOME="$1"
fi

echo $GREEN"Installing cmake, libusb, eigen3, boost, opencv, glfw3, glew, gtest."$WHITE
sudo apt update
sudo apt -y install apt-utils build-essential software-properties-common wget unzip curl
sudo apt -y install cmake libopencv-dev git # CMake, OpenCV, and Git
sudo apt -y install libusb-1.0-0-dev libeigen3-dev # Boost, Libusb and eigen3
sudo apt -y install libglew-dev libglfw3-dev
sudo apt -y install libgtest-dev  # gtest

echo $GREEN"Would you like to install the Python bindings and libraries(pybind11, numpy, pytest)?[y/n]"$WHITE
read PYBINDINGS
if [ "$PYBINDINGS" = "y" ]; then
    sudo apt -y install python3-pip python3-distutils  # the python package installer
    python3 -m pip install pip --upgrade  # upgrade pip
    python3 -m pip install pytest numpy
    cd $openebRoot/..
    wget https://github.com/pybind/pybind11/archive/v$pybind_version.zip
    unzip v$pybind_version.zip
    cd pybind11-$pybind_version/
    mkdir build && cd build
    cmake .. -DPYBIND11_TEST=OFF
    cmake --build .
    sudo cmake --build . --target "$PYBIND_HOME"
else
    echo $RED"The Python bindings will not be installed."$WHITE
fi

cd $openebRoot
echo $GREEN"Generating the openEB build."$WHITE
mkdir build ; cd build
if [ "$PYBINDINGS" != "y" ]; then
    cmake .. -DBUILD_TESTING=OFF -DCOMPILE_PYTHON3_BINDINGS=OFF
else
    cmake .. -DBUILD_TESTING=OFF
fi
echo $GREEN"Building openEB."$WHITE
cmake --build . --config Release --parallel 4