# OpenEB MacOS Build, Test, and Installation Guide
**In this guide**
- [Tested Configurations](#tested-configurations)
- [Build OpenEB C++ Binaries from Source](#build-and-install-openeb-binaries)
- [Install Python API Requirements](#install-python-api-requirements)
- [Upgrade OpenEB](#upgrade-openeb)

## Tested Configurations
| Component | Configuration |
|-----------|----------|
| **OS** | Monterey |
| **Architecture** | x64 |
| **CPU** | with AVX2 support |
| **GPU** | with OpenGL +3.0 support
| **[Python3.8](https://www.python.org/downloads/release/python-3819/)**</br>**[Python3.9](https://www.python.org/downloads/release/python-3919/)** | Ubuntu 20.04 64-bit |
| **[Python3.9](https://www.python.org/downloads/release/python-3919/)**</br>**[Python3.10](https://www.python.org/downloads/release/python-31014/)** | Ubuntu 22.04 64-bit |

Building OpenEB with component configurations other than what is listed above may require adjustments to this guide, or to the code itself.

## Software Requirements
This guide uses [Homebrew](https://brew.sh/) formulae to retrieve packages. It also requires that you have a working [C++ compiler](https://developer.apple.com/xcode/cpp/).
1. Run the command below to make sure you have the appropriate resources to use the build OpenEB binaries on your machine.
    ```console
    brew install \
        boost \
        cmake \
        eigen \
        ffmpeg \
        git \
        glew \
        glfw \
        hdf5 \
        opencv \
        protobuf \
        libusb
    ```
1. If you'll be using the Python API, Python bindings for C++, or and of the test applications, install Python3 as well.
    ```console
    brew install python3
    ```

    > **Note**: *For externally managed environments, activate a Python virtual environment and use it for subsequent installation and build instructions.*
    - Navigate to your development area and activate a Python virtual environment.
        ```console
        python3 -m venv .venv && \
        source .venv/bin/activate
        ```
1. Retrieve the latest [OpenEB source](https://github.com/prophesee-ai/openeb), or your desired branch, and navigate into the project folder.
    ```console
    git clone https://github.com/prophesee-ai/openeb.git && cd openeb
    ```
1. Once inside your OpenEB directory, create an environment variable to reference that location.
    ```console
    export OPENEB_SRC_DIR=`pwd`
    ```
## Build OpenEB C++ Binaries from Source
### Initial Setup and Build Configuration
1. Follow the [Software Requirements](#software-requirements) section to setup your machine and retrieve the OpenEB source code.
1. Create a build folder for the project files and navigate into the folder.
    ```console
    mkdir build && cd build
    ```
1. Use CMake to create the build files for OpenEB.
    ```console
    cmake ..
    ```

#### Additional Build Configurations
When you build OpenEB you have the option to:
1. Enable/Disable test applications
1. Enable/Disable Python binding for C++
1. Change the build type (`Release`, `Debug`, ect.)

By default, test applications and Python C++ bindings are disabled. However, you can control them by using the CMake configuration options given in the table below.

| Build Description | CMake Configuration Option |
|-------------------|----------------------------|
| **Enable Test Applications** | `-DBUILD_TESTING=ON` |
| **Enable Python Bindings** | `-DCOMPILE_PYTHON3_BINDINGS=ON` |
| **Change the build type** | `-DCMAKE_BUILD_TYPE=<Release\|Debug>`|

Once you've completed the requirements for
any of the additional build configurations, use the option flags individually, or together, to enable the features you desire.

##### Test Applications
1. Complete steps 1&rarr;4 of the [initial setup and build configuration.](#initial-setup-and-build-configuration)
1. Install the Google Test and `pytest` dependencies.
    ```console
    brew install googletest && \
    pip3 install pytest
    ```
1. Download the test files and place them in your OpenEB source directory using the command below.
    ```console
    wget https://kdrive.infomaniak.com/2/app/975517/share/cddcc78a-3480-420f-bc19-17d5b0535ca4/archive/1aa2f344-7856-4f29-bd6a-21d7d78762bd/download -O /tmp/open-test-dataset.zip && \
    unzip /tmp/open-test-dataset.zip -d ${OPENEB_SRC_DIR}/datasets &&\
    rm /tmp/open-test-dataset.zip
    ```
    > **NOTE**: *The total download file size is ~1.4GB.*
1. Enable the test applications using the CMake command and options below.
    ```console
    cmake --fresh .. -DBUILD_TESTING=ON
    ```

##### Python Bindings for C++
1. Complete steps 1&rarr;4 for the [initial setup and build configuration.](#initial-setup-and-build-configuration)
1. Install pybind11.
    ```console
    brew install pybind11
    ```
1. Enable the Python bindings using the CMake command and options below.
    ```console
    cmake --fresh .. -DCOMPILE_PYTHON3_BINDINGS=ON
    ```

### Build, Test, and Install OpenEB Binaries
1. Follow the step in the [Initial Setup and Build Configuration](#initial-setup-and-build-configuration) section to create the build files that best suite your needs.
1. Build OpenEB on your machine.
    ```console
    cmake --build . -j4
    ```
1. **[Optional]** If testing was enabled in your build, you can now run the test suite.
    ```console
    ctest --verbose
    ```
1. Install OpenEB.
    - Use CMake to make the system path the installation target.
        ```console
        sudo cmake --build . --target install
        ```
    - Set an alternative installation target.
        ```console
        sudo cmake --build . --target install -DCMAKE_INSTALL_PREFIX=<OPENEB_INSTALL_DIR>
        ```
1. Update the necessary environment variables.
    ```console
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/local/lib/hdf5/plugin  # On Ubuntu 20.04
    export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/local/hdf5/lib/plugin  # On Ubuntu 22.04
    ```

## Install Python API Requirements
> **Note**: *Use of the Python API requires specific versions of Python. Refer to the [Tested Configurations](#tested-configurations) table to determine the version of Python that best suites your application needs.*

1. Follow the [Software Requirements](#software-requirements) section to setup your machine and retrieve the OpenEB source code.
1. Install the additional library dependencies using the requirements file.
    ```console
    pip3 install -r ${OPENEB_SRC_DIR}/utils/python/requirements.txt
    ```
### Python API Machine Learning Requirements
Additional hardware, third-party software, and Python packages are required for using OpenEB machine learning (ML) features.
1. Follow all steps for [installing the Python API requirements](#install-python-api-requirements).
1. Ensure that your machine has, or is capable of running an Nvidia GPU that supports:
    - [CUDA +11.6](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/overview.html)

1. Install the required [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/overview.html) components once the hardware requirements are met.
1. Install [PyTorch +1.13.1](https://pytorch.org/get-started/locally/).
1. Install the additional Python packages for ML features.
    ```console
    pip3 install -r ${OPENEB_SRC_DIR}/utils/python/ml-requirements.txt
    ```
    > **Note**: *If you are using an externally managed environment, install the Python packages within your virtual environment*


## Upgrade OpenEB
To upgrade and existing installation of OpenEB:
1. Refer to our [Release Notes](https://docs.prophesee.ai/stable/release_notes.html) before performing any upgrades to your existing OpenEB installation.
<quote>This will help you navigate changes such as API and firmware updates that can impact applications using the Metavision SDK and event-cameras.</quote>
1. Remove the previously installed OpenEB software.
    ```console
    sudo make uninstall
    ```
1. Remove any remaining OpenEB files from your system's `PATH` and other environment variable locations (`PYTHONPATH` and `LD_LIBRARY_PATH`).
1. Follow the instructions to [build](#build-openeb-c-binaries-from-source) and [install](#install-openeb) OpenEB.

