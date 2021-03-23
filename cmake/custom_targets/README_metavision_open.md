# Metavision: installation from source

This page describes how to compile and install the **OpenEB** codebase.
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
sudo cp *.a /usr/lib
```

The Python bindings rely on the [pybind11](https://github.com/pybind) library, specifically version 2.4.3.

Note: pybind11 is required only if you use the python interface; you can opt out of creating these bindings by passing the argument ` -DCOMPILE_PYTHON3_BINDINGS=OFF` at step 3 during compilation (see below); you will not need to install pybind11, but you won't be able to use our python interface.

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

### Metavision compilation

 1. Retrieve the code `git clone https://github.com/prophesee-ai/openeb.git`
 2. Create and open the build directory, where temporary file will be created: `mkdir build && cd build`
 3. Generate the makefiles using CMake: `cmake .. -DBUILD_TESTING=OFF`
 4. Compile: `cmake --build . --config Release -- -j 4`
 5. Install: `sudo cmake --build . --target install`
 6. Add /usr/local/lib to LD_LIBRARY_PATH:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

Note that you can add this command to your ~/.bashrc or ~/.zshrc to make it permanent. 

 7. If you are planning to work with Prophesee cameras or data recordings, then install **Prophesee plugins** as described in [online documentation](https://docs.prophesee.ai/).

### (Optional) Running the test suite


Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

*Note* that [the Metavision Hal Plugins](https://docs.prophesee.ai/2.2.0/installation/linux.html#camera-plugins) must be installed for most of these tests to run.

 * Go to [this page](https://dataset.prophesee.ai/index.php/s/ozjYOAAKTUshudQ) to download the files necessary to run the
tests. The obtained archive weighs around 3 Gb.

 * Extract and put the content of this archive to <METAVISION_SRC_DIR>/datasets/

 * Regenerate the makefiles with the test options on.

  ```bash
  cd <METAVISION_SRC_DIR>/build
  cmake .. -DBUILD_TESTING=ON
  ```

 * Compile again.  `cmake --build . --config Release -- -j 4`

 * Finally, run the test suite:   `ctest --verbose`

## Compiling on Windows

### Prerequisites

To compile Metavision, you will need to install some extra tools:

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

#### Install Python 3.7 or 3.8

* Download "Windows x86-64 executable installer" for one of these Python versions:
  * [Python 3.7](https://www.python.org/downloads/release/python-379/)
  * [Python 3.8](https://www.python.org/downloads/release/python-388/)
* Run the installer and follow the prompt. We advise you to check the box that propose to update the PATH or to update it manually with this command, replacing "Username" with your own.

```bash
C:\Users\Username\AppData\Local\Programs\Python\Python37;C:\Users\Username\AppData\Local\Programs\Python\Python37\Scripts (assuming the default install path was used)
````

* Edit your environment variable ``PYTHONPATH`` and append the following path (if it is not already there):

```bash
"C:\Program Files\Prophesee\lib\python3\site-packages"
```

* Re-start your session
* Finally, install additionally required Python packages using pip:
```bash
python3 -m pip install pip --upgrade
python3 -m pip install "numpy==1.19.5" pytest
```

### Metavision compilation

First, retrieve the codebase:

```bash
git clone https://github.com/prophesee-ai/openeb.git
```

#### Compilation using CMake

Open a command prompt inside the ``openeb`` folder and do as follows:

 1. Create and open the build directory, where temporary files will be created: `mkdir build && cd build`
 2. Generate the build using CMake: `cmake -A x64 -DCMAKE_TOOLCHAIN_FILE=<VCPKG_SRC_DIR>\scripts\buildsystems\vcpkg.cmake ..`
 3. Compile: `cmake --build . --config Release --parallel 4`

#### Compilation using MS Visual Studio

Open a command prompt inside the ``openeb`` folder and do as follows:

 1. Create and open the build directory, where temporary files will be created: `mkdir build && cd build`
 2. Generate the Visual Studio files using CMake: `cmake  -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=<VCPKG_SRC_DIR>\scripts\buildsystems\vcpkg.cmake ..` (adapt to your Visual Studio version)
 3. Open the solution file `metavision.sln`, select the `Release` configuration and build the `ALL_BUILD` project.

#### Installing Prophesee Plugins

Install **Prophesee plugins** if you plan to work with Prophesee cameras or data recordings:

* Go to the [Metavision sign-up page](https://www.prophesee.ai/metavision-intelligence-plugins-download),
  download ``Metavision_HAL_Prophesee_plugins_220_Setup.exe`` and run it
* If an alert message from Windows pops up during install, click *more info*, then *run anyway*

### (Optional) Running the test suite


Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

*Note* that [the Metavision Hal Plugins](https://docs.prophesee.ai/2.2.0/installation/windows.html#camera-plugins) must be installed for most of these tests to run.

 * Go to [this page](https://dataset.prophesee.ai/index.php/s/ozjYOAAKTUshudQ) to download the files necessary to run the tests.
 Click ``Download`` on the top right folder. The obtained archive weighs around 3 Gb.
 * Extract and put the content of this archive to ``<METAVISION_SRC_DIR>/datasets/``

 * To run the test suite you need to reconfigure your build environment using CMake and to recompile


   * Compilation using CMake

    1. Regenerate the build using CMake:

        ```
        cd <METAVISION_SRC_DIR>/build
        cmake -A x64 -DCMAKE_TOOLCHAIN_FILE=<VCPKG_SRC_DIR>\scripts\buildsystems\vcpkg.cmake -DBUILD_TESTING=ON ..
        ```
    2. Compile    `cmake --build . --config Release --parallel 4`


   * Compilation using MS Visual Studio

    1. Generate the Visual Studio files using CMake (adapt the command to your Visual Studio version):

        `cmake  -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=<VCPKG_SRC_DIR>\scripts\buildsystems\vcpkg.cmake -DBUILD_TESTING=ON ..``

    2. Open the solution file `metavision.sln`, select the `Release` configuration and build the `ALL_BUILD` project.

 * Running the test suite is then simply `ctest -C Release`
