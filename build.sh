#!/bin/bash
set -ex

# Create and enter build directory
mkdir -p build
cd build

# Configure CMake
cmake ${CMAKE_ARGS} \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SAMPLES=ON \
    -DCOMPILE_PYTHON3_BINDINGS=ON \
    -DBUILD_TESTING=OFF \
    -DCODE_COVERAGE=OFF \
    -DGENERATE_DOC=OFF \
    -DPython3_EXECUTABLE="$PYTHON" \
    -DUDEV_RULES_SYSTEM_INSTALL=OFF \
    ..

# Build and install
cmake --build . --target install -j${CPU_COUNT}
