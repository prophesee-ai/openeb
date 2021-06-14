# OpenEB

OpenEB enables anyone to get a better understanding of Event-Based Vision, directly interact with events and build
their own applications or plugins. As a camera manufacturer, ensure your customers benefit from the most advanced 
Event-Based software suite available by building your own plugin. As a creator, scientist, academic, join and contribute
to the fast-growing Event-Based Vision community.

OpenEB is composed of 5 fundamental software modules:
* HAL: Hardware abstraction layer allowing Metavision Intelligence Suite to operate with any Event-Based Vision device.
* Base: Foundations and common definitions of Event-Based applications.
* Core: Generic algorithms for visualization, event stream manipulation, applicative pipeline generation.
* Driver: High-level abstraction to easily interact with Event-Based cameras.
* UI: Viewer and display controllers for Event-Based data.

This document describes how to compile and install the **OpenEB** codebase.
For more information, refer to our [online documentation](https://docs.prophesee.ai/).

## Compiling on Ubuntu

Currently, we support Ubuntu 18.04 and 20.04.

### Prerequisites

Install the following dependencies:

```bash
sudo apt update
sudo apt -y install apt-utils build-essential software-properties-common wget unzip curl
sudo apt -y install cmake  libopencv-dev git # CMake, OpenCV, and Git
sudo apt -y install libboost-all-dev libusb-1.0-0-dev libeigen3-dev # Boost, Libusb and eigen3
sudo apt -y install libglew-dev libglfw3-dev
sudo apt -y install libgtest-dev  # gtest
sudo apt -y install python3-pip python3-distutils  # the python package installer
python3 -m pip install pip --upgrade  # upgrade pip
python3 -m pip install pytest "numpy==1.19.5"
```

If you want to run tests, then you need to compile **gtest** package (this is optional):

```bash
cd /usr/src/gtest
sudo cmake .
sudo make
sudo make install
```

The Python bindings rely on the [pybind11](https://github.com/pybind) library, specifically version 2.4.3.

Note: pybind11 is required only if you use the python API.
You can opt out of creating these bindings by passing the argument ` -DCOMPILE_PYTHON3_BINDINGS=OFF` at step 3 during compilation (see below).
In that case, you will not need to install pybind11, but you won't be able to use our python interface.

Unfortunately, there is no pre-compiled version of pybind11 available, so you need to install it manually:

```bash
wget https://github.com/pybind/pybind11/archive/v2.4.3.zip
unzip v2.4.3.zip
cd pybind11-2.4.3/
mkdir build && cd build
cmake .. -DPYBIND11_TEST=OFF
cmake --build .
sudo cmake --build . --target install
```

### Compilation

 1. Retrieve the code `git clone https://github.com/prophesee-ai/openeb.git`
 2. Create and open the build directory in the `openeb` folder: `cd openeb; mkdir build && cd build`
 3. Generate the makefiles using CMake: `cmake .. -DBUILD_TESTING=OFF`
 4. Compile: `cmake --build . --config Release -- -j 4`
 
You can now use OpenEB directly from the build folder.
For this, you will need to update your environment variables using this script:

```bash
source ./utils/scripts/setup_env.sh
```

Optionally, you can deploy the OpenEB files in the system path to use them as 3rd party dependency in some other code
with the following command: `sudo cmake --build . --target install`. In that case, you will also need to update 
`LD_LIBRARY_PATH` with `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib` (If you want to update this path
permanently, you should add the previous command in your ~/.bashrc)

If you are planning to work with Prophesee cameras or data recordings, then install **Prophesee plugins**:

 * Go to the [sign-up page for Prophesee Camera Plugins](https://www.prophesee.ai/metavision-intelligence-plugins-download/)
 * Download the `.list` file for your version of Ubuntu and add it to the folder `/etc/apt/sources.list.d`
 * Install Prophesee plugins:

```sudo apt update
sudo apt install metavision-hal-prophesee-plugins
```

### (Optional) Running the test suite


Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

*Note* that the [Prophesee Camera Plugins](https://www.prophesee.ai/metavision-intelligence-plugins-download/) must be installed for most of these tests to run.

 * Download [the files](https://dataset.prophesee.ai/index.php/s/ozjYOAAKTUshudQ) necessary to run the tests.
   Click `Download` on the top right folder. The obtained archive weighs around 3 Gb.

 * Extract and put the content of this archive to `<OPENEB_SRC_DIR>/datasets/`

 * Regenerate the makefiles with the test options on.

  ```bash
  cd <OPENEB_SRC_DIR>/build
  cmake .. -DBUILD_TESTING=ON
  ```

 * Compile again.  `cmake --build . --config Release -- -j 4`

 * Finally, run the test suite:   `ctest --verbose`

## Compiling on Windows

### Prerequisites

To compile OpenEB, you will need to install some extra tools:

 * install cmake
 * install the MS Visual Studio IDE to get a C++ compiler.
    If you only want the compiler, you can install C++ Build Tools instead.
 * install [vcpkg](https://github.com/microsoft/vcpkg) that will be used for installing dependencies (we recommend the specific version here):
    * `cd <VCPKG_SRC_DIR>`
	  * `git clone https://github.com/microsoft/vcpkg.git`
    * `git checkout 08c951fef9de63cde1c6b94245a63db826be2e32`
	  * `cd vcpkg && bootstrap-vcpkg.bat`
    * `setx VCPKG_DEFAULT_TRIPLET x64-windows`, this will inform vcpkg to install packages for an x64-windows target. You need to close the command line and re-open it to ensure that this variable is set
  * finally, install the libraries by running `vcpkg.exe install libusb eigen3 boost opencv glfw3 glew gtest`

You can extract the tar archives using [7zip](https://www.7-zip.org/) or a similar tool.

#### Install pybind

The Python bindings rely on the [pybind11](https://github.com/pybind) library.
You can install pybind using vcpkg: `vcpkg.exe install pybind11`

Note: pybind11 is required only if you use the python API. 
You can opt out of creating these bindings by passing the argument ` -DCOMPILE_PYTHON3_BINDINGS=OFF` at step 2 during compilation (see section "Compilation using CMake").
In that case, you will not need to install pybind11, but you won't be able to use our python interface.


#### Install Python 3.7 or 3.8

* Download "Windows x86-64 executable installer" for one of these Python versions:
  * [Python 3.7](https://www.python.org/downloads/release/python-3710/)
  * [Python 3.8](https://www.python.org/downloads/release/python-389/)
* Run the installer and follow the prompt. We advise you to check the box that propose to update the PATH or to update it manually with this command, replacing "Username" with your own.

```bash
C:\Users\Username\AppData\Local\Programs\Python\Python37;C:\Users\Username\AppData\Local\Programs\Python\Python37\Scripts (assuming the default install path was used)
````

* Edit your environment variable `PYTHONPATH` and append the following path (if it is not already there):

```bash
"C:\Program Files\Prophesee\lib\python3\site-packages"
```

* Re-start your session
* Finally, install additionally required Python packages using pip:
```bash
python3 -m pip install pip --upgrade
python3 -m pip install "numpy==1.19.5" pytest
```

### Compilation

First, retrieve the codebase:

```bash
git clone https://github.com/prophesee-ai/openeb.git
```

#### Compilation using CMake

Open a command prompt inside the `openeb` folder and do as follows:

 1. Create and open the build directory, where temporary files will be created: `mkdir build && cd build`
 2. Generate the build using CMake: `cmake -A x64 -DCMAKE_TOOLCHAIN_FILE=<VCPKG_SRC_DIR>\scripts\buildsystems\vcpkg.cmake ..`
 3. Compile: `cmake --build . --config Release --parallel 4`
 
You can now use OpenEB directly from the build folder.
For this, you will need to update your environment variables using this script:

```bash
.\utils\scripts\setup_env.bat
```

Optionally, you can deploy the OpenEB files in the system path to use them as 3rd party dependency in some other code
with the following command: `sudo cmake --build . --target install`. 

#### Compilation using MS Visual Studio

Open a command prompt inside the `openeb` folder and do as follows:

 1. Create and open the build directory, where temporary files will be created: `mkdir build && cd build`
 2. Generate the Visual Studio files using CMake: `cmake  -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=<VCPKG_SRC_DIR>\scripts\buildsystems\vcpkg.cmake ..` (adapt to your Visual Studio version)
 3. Open the solution file `metavision.sln`, select the `Release` configuration and build the `ALL_BUILD` project.

#### Installing Prophesee Plugins

Install **Prophesee plugins** if you plan to work with Prophesee cameras or data recordings:

 * Go to the [sign-up page for Prophesee Camera Plugins](https://www.prophesee.ai/metavision-intelligence-plugins-download/)
 * Follow the Camera Plugins download link provided after sign-up
 * Among the list of Camera Plugins installers, download the one with the version number matching your OpenEB version
 * Run the installer

### (Optional) Running the test suite


Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

*Note* that the [Prophesee Camera Plugins](https://www.prophesee.ai/metavision-intelligence-plugins-download/) must be installed for most of these tests to run.

 * Download [the files](https://dataset.prophesee.ai/index.php/s/ozjYOAAKTUshudQ) necessary to run the tests.
   Click `Download` on the top right folder. The obtained archive weighs around 3 Gb.
 * Extract and put the content of this archive to `<OPENEB_SRC_DIR>/datasets/`

 * To run the test suite you need to reconfigure your build environment using CMake and to recompile


   * Compilation using CMake

    1. Regenerate the build using CMake:

        ```
        cd <OPENEB_SRC_DIR>/build
        cmake -A x64 -DCMAKE_TOOLCHAIN_FILE=<VCPKG_SRC_DIR>\scripts\buildsystems\vcpkg.cmake -DBUILD_TESTING=ON ..
        ```
    2. Compile    `cmake --build . --config Release --parallel 4`


   * Compilation using MS Visual Studio

    1. Generate the Visual Studio files using CMake (adapt the command to your Visual Studio version):

        `cmake  -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=<VCPKG_SRC_DIR>\scripts\buildsystems\vcpkg.cmake -DBUILD_TESTING=ON ..`

    2. Open the solution file `metavision.sln`, select the `Release` configuration and build the `ALL_BUILD` project.

 * Running the test suite is then simply `ctest -C Release`
