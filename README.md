# OpenEB

OpenEB is the open source project associated with [Metavision SDK](https://www.prophesee.ai/metavision-intelligence/)

It enables anyone to get a better understanding of event-based vision, directly interact with events and build
their own applications or plugins. As a camera manufacturer, ensure your customers benefit from the most advanced 
event-based software suite available by building your own plugin. As a creator, scientist, academic, join and contribute
to the fast-growing event-based vision community.

OpenEB is composed of the Open modules of Metavision SDK:
* HAL: Hardware Abstraction Layer to operate any event-based vision device.
* Base: Foundations and common definitions of event-based applications.
* Core: Generic algorithms for visualization, event stream manipulation, applicative pipeline generation.
* Core ML: Generic functions for Machine Learning, event_to_video and video_to_event pipelines.
* Driver: High-level abstraction built on the top of HAL to easily interact with event-based cameras.
* UI: Viewer and display controllers for event-based data.

OpenEB also contains the source code of Prophesee camera plugins, enabling to stream data from our event-based cameras
and to read recordings of event-based data. The supported cameras are:
* EVK2 - Gen4.1 HD
* EVK3 - Gen 3.1 VGA / Gen4.1 HD
* EVK4 - HD

This document describes how to compile and install the OpenEB codebase.
For further information, refer to our [online documentation](https://docs.prophesee.ai/) where you will find
some [tutorials](https://docs.prophesee.ai/stable/metavision_sdk/tutorials/index.html) to get you started in C++ or Python,
some [samples](https://docs.prophesee.ai/stable/samples.html) to discover how to use
[our API](https://docs.prophesee.ai/stable/api.html) and a more detailed
[description of our modules and packaging](https://docs.prophesee.ai/stable/modules.html).


## Compiling on Linux

Compilation and execution were tested on platforms that meet the following requirements:

  * Linux: Ubuntu 20.04 or 22.04 64-bit
  * Architecture: amd64 (a.k.a. x64)
  * Graphic card with support of OpenGL 3.0 minimum
  * CPU with [support of AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#CPUs_with_AVX2)

Compilation on other platforms (alternate Linux distributions, different versions of Ubuntu, ARM processor architecture etc.)
was not tested. For those platforms some adjustments to this guide or to the code itself may be required.


### Upgrading OpenEB

If you are upgrading OpenEB from a previous version, you should first read carefully the [Release Notes](https://docs.prophesee.ai/stable/release_notes.html)
as some changes may impact your usage of our SDK (e.g. API updates) and cameras (e.g. [firmware update](https://support.prophesee.ai/portal/en/kb/articles/evk-firmware-versions) might be necessary).

Then, you need to clean your system from previously installed Prophesee software. If after a previous compilation, you chose to
deploy the Metavision files in your system path, then go to the `build` folder in the source code directory and
launch the following command to remove those files:

```bash
sudo make uninstall
```

In addition, make a global check in your system paths (`/usr/lib`, `/usr/local/lib`, `/usr/include`, `/usr/local/include`)
and in your environment variables (`PATH`, `PYTHONPATH` and `LD_LIBRARY_PATH`) to remove occurrences of Prophesee or Metavision files.

### Prerequisites

Install the following dependencies:

```bash
sudo apt update
sudo apt -y install apt-utils build-essential software-properties-common wget unzip curl git cmake
sudo apt -y install libopencv-dev libboost-all-dev libusb-1.0-0-dev
sudo apt -y install libhdf5-dev hdf5-tools libglew-dev libglfw3-dev libcanberra-gtk-module ffmpeg 
```

Optionally, if you want to run the tests, you need to install Google Gtest and Gmock packages.
For more details, see [Google Test User Guide](https://google.github.io/googletest/):

```bash
sudo apt -y install libgtest-dev libgmock-dev
```

For the Python API, you will need Python and some additional libraries.
If Python is not available on your system, install it
We support Python 3.8 and 3.9 on Ubuntu 20.04 and Python 3.9 and 3.10 on Ubuntu 22.04.
If you want to use other versions of Python, some source code modifications will be necessary

Then install `pip` and some Python libraries:
```bash
sudo apt -y install python3-pip python3-distutils
sudo apt -y install python3.X-dev  # where X is 8, 9 or 10 depending on your Python version (3.8, 3.9 or 3.10)
python3 -m pip install pip --upgrade
python3 -m pip install "opencv-python==4.5.5.64" "sk-video==1.1.10" "fire==0.4.0" "numpy==1.23.4" pandas scipy h5py
python3 -m pip install jupyter jupyterlab matplotlib "ipywidgets==7.6.5" pytest command_runner
```

The Python bindings of the C++ API rely on the [pybind11](https://github.com/pybind) library, specifically version 2.6.0.

*Note* that pybind11 is required only if you want to use the Python bindings of the C++ API .
You can opt out of creating these bindings by passing the argument `-DCOMPILE_PYTHON3_BINDINGS=OFF` at step 3 during compilation (see below).
In that case, you will not need to install pybind11, but you won't be able to use our Python interface to the C++ API.

Unfortunately, there is no pre-compiled version of pybind11 available, so you need to install it manually:

```bash
wget https://github.com/pybind/pybind11/archive/v2.6.0.zip
unzip v2.6.0.zip
cd pybind11-2.6.0/
mkdir build && cd build
cmake .. -DPYBIND11_TEST=OFF
cmake --build .
sudo cmake --build . --target install
```

To use Machine Learning features, you need to install some additional dependencies.

First, if you have some Nvidia hardware with GPUs, you can optionally install [CUDA (11.6 or 11.7)](https://developer.nvidia.com/cuda-downloads)
and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) to leverage them with pytorch and libtorch.

Make sure that you install a version of CUDA that is compatible with your GPUs by checking
[Nvidia compatibility page](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html).

Note that, at the moment, we don't support [OpenCL](https://www.khronos.org/opencl/) and AMD GPUs.

Then, you will need to install [PyTorch 1.13.1](https://pytorch.org/get-started/locally/).
Retrieve and execute the pip command from the installation guide.
If the latest Pytorch version doesn't match, please consider looking  into the [previous versions section](<https://pytorch.org/get-started/previous-versions/>).

Then install some extra Python libraries:

```bash
python3 -m pip install "numba==0.56.3" "profilehooks==1.12.0" "pytorch_lightning==1.8.6" "tqdm==4.63.0" "kornia==0.6.8"
```


### Compilation

 1. Retrieve the code `git clone https://github.com/prophesee-ai/openeb.git --branch 4.2.1`.
    (If you choose to download an archive of OpenEB from GitHub rather than cloning the repository,
    you need to ensure that you select a ``Full.Source.Code.*`` archive instead of using
    the automatically generated ``Source.Code.*`` archives. This is because the latter do not include
    a necessary submodule.)
 2. Create and open the build directory in the `openeb` folder (absolute path to this directory is called `OPENEB_SRC_DIR` in next sections): `cd openeb; mkdir build && cd build`
 3. Generate the makefiles using CMake: `cmake .. -DBUILD_TESTING=OFF`.
    If you want to specify to cmake which version of Python to consider, you should use the option ``-DPython3_EXECUTABLE=<path_to_python_to_use>``.
    This is useful, for example, when you have a more recent version of Python than the ones we support installed on your system.
    In that case, cmake would select it and compilation might fail.
 4. Compile: `cmake --build . --config Release -- -j 4`

Once the compilation is finished, you have two options: you can choose to work directly from the `build` folder
or you can deploy the OpenEB files in the system path (`/usr/local/lib`, `/usr/local/include`...).

* Option 1 - working from `build` folder

  * To use OpenEB directly from the `build` folder, you need to update some environment variables using this script
    (which you may add to your `~/.bashrc` to make it permanent):

    ```bash
    source utils/scripts/setup_env.sh
    ```

  * Prophesee camera plugins are included in OpenEB, but you still need to copy the udev rules files in the system path
    and reload them so that your camera is detected with this command:

    ```bash
    sudo cp <OPENEB_SRC_DIR>/hal_psee_plugins/resources/rules/*.rules /etc/udev/rules.d
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ```

* Option 2 - deploying in the system path

  * To deploy OpenEB, launch the following command:

    ```bash
    sudo cmake --build . --target install
    ```

    Note that you ou can also deploy the OpenEB files (applications, samples, libraries etc.) in a directory of your choice by using
    the `CMAKE_INSTALL_PREFIX` variable (`-DCMAKE_INSTALL_PREFIX=<OPENEB_INSTALL_DIR>`) when generating the makefiles
    in step 3. Similarly, you can configure the directory where the Python packages will be deployed using the
    `PYTHON3_SITE_PACKAGES` variable (`-DPYTHON3_SITE_PACKAGES=<PYTHON3_PACKAGES_INSTALL_DIR>`).

  * you also need to update `LD_LIBRARY_PATH` and `HDF5_PLUGIN_PATH`
    (which you may add to your `~/.bashrc` to make it permanent):

    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/local/hdf5/lib/plugin
    ```

Note that if you are using a third-party camera, you need to install the plugin provided
by the camera vendor and specify the location of the plugin using the `MV_HAL_PLUGIN_PATH` environment variable.

To get started with OpenEB, you can download some [sample recordings](https://docs.prophesee.ai/stable/datasets.html) 
and visualize them with [metavision_viewer](https://docs.prophesee.ai/stable/metavision_sdk/modules/driver/samples/viewer.html)
or you can stream data from your Prophesee-compatible event-based camera.

### Running the test suite (Optional)

Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

 * Download [the files](https://dataset.prophesee.ai/index.php/s/tiP0wl0r5aW5efL) necessary to run the tests.
   Click `Download` on the top right folder. Beware of the size of the obtained archive which weighs around 1.2 Gb.

 * Extract and put the content of this archive to `<OPENEB_SRC_DIR>/`. For instance, the correct path of sequence `gen31_timer.raw` should be `<OPENEB_SRC_DIR>/datasets/openeb/gen31_timer.raw`.

 * Regenerate the makefiles with the test options on.

  ```bash
  cd <OPENEB_SRC_DIR>/build
  cmake .. -DBUILD_TESTING=ON
  ```

 * Compile again.  `cmake --build . --config Release -- -j 4`

 * Finally, run the test suite:   `ctest --verbose`

## Compiling on Windows

Currently, we support only Windows 10. 
Compilation on other versions of Windows was not tested.
For those platforms some adjustments to this guide or to the code itself may be required.

### Upgrading OpenEB

If you are upgrading OpenEB from a previous version, you should first read carefully the [Release Notes](https://docs.prophesee.ai/stable/release_notes.html)
as some changes may impact your usage of our SDK (e.g. :API updates) and cameras (e.g. [firmware update](https://support.prophesee.ai/portal/en/kb/articles/evk-firmware-versions) might be necessary).

Then, if you have previously installed any Prophesee's software, you will need to uninstall it first.
Remove the folders where you installed Metavision artifacts (check both the `build` folder of the source code and
`C:\Program Files\Prophesee` which is the default install path of the deployment step).

### Prerequisites

Some steps of this procedure don't work on FAT32 and exFAT file system.
Hence, make sure that you are using a NTFS file system before going further.

You must enable the support for long paths:
 * Hit the Windows key, type gpedit.msc and press Enter
 * Navigate to Local Computer Policy > Computer Configuration > Administrative Templates > System > Filesystem
 * Double-click the "Enable Win32 long paths" option, select the "Enabled" option and click "OK"

To compile OpenEB, you will need to install some extra tools:

 * install [git](https://git-scm.com/download/win)
 * install [CMake 3.21](https://cmake.org/files/v3.21/cmake-3.21.7-windows-x86_64.msi)
 * install Microsoft C++ compiler (64-bit). You can choose one of the following solutions:
    * For building only, you can install MS Build Tools (free, part of Windows 10 SDK package)
      * Download and run ["Build tools for Visual Studio 2022" installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
      * Select "C++ build tools", make sure Windows 10 SDK is checked, and add English Language Pack
    * For development, you can also download and run [Visual Studio Installer](https://visualstudio.microsoft.com/downloads/)    
 * install [vcpkg](https://github.com/microsoft/vcpkg) that will be used for installing dependencies:
    * download and extract [vcpkg version 2022.03.10](https://github.com/microsoft/vcpkg/archive/refs/tags/2022.03.10.zip)
    * `cd <VCPKG_SRC_DIR>`
    * `bootstrap-vcpkg.bat`
  * install the libraries by running `vcpkg.exe install --triplet x64-windows libusb eigen3 boost opencv glfw3 glew gtest dirent hdf5[cpp,threadsafe,tools,zlib]`
    * Note that to avoid using `--triplet x64-windows`, which informs vcpkg to install packages for a x64-windows target,
      you can run `setx VCPKG_DEFAULT_TRIPLET x64-windows` (you need to close the command line and re-open it to ensure that this variable is set)
  * Finally, download and install [ffmpeg](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z) and add the `bin` directory to your PATH.

Note that if you are using vcpkg for various projects or multiple versions of OpenEB, you might want to optimize the
number of vcpkg install you manage. To do so, you will need the versions of the libraries we require.
Those can be found in the [vcpkg repository](https://github.com/microsoft/vcpkg/tree/2022.03.10/versions) but we list them here for convenience:
  * libusb: 1.0.24
  * eigen3: 3.4.0
  * boost: 1.78.0
  * opencv: 4.5.5
  * glfw3: 3.3.6
  * glew: 2.2.0
  * gtest: 1.11.0
  * dirent: 1.23.2
  * hdf5: 1.12.1

#### Installing Python and libraries

* Download "Windows x86-64 executable installer" for one of these Python versions:
  * [Python 3.8](https://www.python.org/downloads/release/python-3810/)
  * [Python 3.9](https://www.python.org/downloads/release/python-3913/)
* Add Python install and script directories in your `PATH` and make sure they are listed before
  the `WindowsApps` folder which contains a Python alias launching the Microsoft Store. So, if you installed
  Python 3.8 in the default path, your user `PATH` should contain those three lines in that order:
  
```bash
%USERPROFILE%\AppData\Local\Programs\Python\Python38
%USERPROFILE%\AppData\Local\Programs\Python\Python38\Scripts
%USERPROFILE%\AppData\Local\Microsoft\WindowsApps
````

Then install `pip` and some Python libraries:

```bash
python -m pip install pip --upgrade
python -m pip install "opencv-python==4.5.5.64" "sk-video==1.1.10" "fire==0.4.0" "numpy==1.23.4" pandas scipy h5py
python -m pip install jupyter jupyterlab matplotlib "ipywidgets==7.6.5" pytest command_runner
```

#### Install pybind

The Python bindings of the C++ API rely on the [pybind11](https://github.com/pybind) library.
You should install pybind using vcpkg in order to get the appropriate version: `vcpkg.exe install --triplet x64-windows pybind11`

*Note* that pybind11 is required only if you plan to use the Python bindings of the C++ API.
You can opt out of creating these bindings by passing the argument `-DCOMPILE_PYTHON3_BINDINGS=OFF` at step 2 during compilation (see section "Compilation using CMake").
In that case, you will not need to install pybind11, but you won't be able to use our Python interface to the C++ API.

#### Prerequisites for the ML module

To use Machine Learning features, you need to install some additional dependencies.

First, if you have some Nvidia hardware with GPUs, you can optionally install [CUDA (11.6 or 11.7)](https://developer.nvidia.com/cuda-downloads)
and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) to leverage them with pytorch and libtorch.

Then, you will need to install [PyTorch 1.13.1](https://pytorch.org/get-started/locally/).
Retrieve and execute the pip command from the installation guide.
If the latest Pytorch version doesn't match, please consider looking  into the [previous versions section](<https://pytorch.org/get-started/previous-versions/>).

Then install some extra Python libraries:

```bash
python -m pip install "numba==0.56.3" "profilehooks==1.12.0" "pytorch_lightning==1.8.6" "tqdm==4.63.0" "kornia==0.6.8"
```

### Compilation

First, retrieve the codebase:

```bash
git clone https://github.com/prophesee-ai/openeb.git --branch 4.2.1
```

Note that if you choose to download an archive of OpenEB from GitHub rather than cloning the repository,
you need to ensure that you select a ``Full.Source.Code.*`` archive instead of using
the automatically generated ``Source.Code.*`` archives. This is because the latter do not include
a necessary submodule.


#### Compilation using CMake

Open a command prompt inside the `openeb` folder (absolute path to this directory is called `OPENEB_SRC_DIR` in next sections) and do as follows:

 1. Create and open the build directory, where temporary files will be created: `mkdir build && cd build`
 2. Generate the makefiles using CMake: `cmake -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> ..`.
    Note that the value passed to the parameter `-DCMAKE_TOOLCHAIN_FILE` must be an absolute path, not a relative one. 
 3. Compile: `cmake --build . --config Release --parallel 4`
 
Once the compilation is done, you have two options: you can choose to work directly from the `build` folder
or you can deploy the OpenEB files (applications, samples, libraries etc.) in a directory of your choice.

* Option 1 - working from `build` folder

  * To use OpenEB directly from the `build` folder,
  you need to update some environment variables using this script:

    ```bash
    utils\scripts\setup_env.bat
    ```
    
* Option 2 - deploying in a directory of your choice

  * To deploy OpenEB, configure the target folder (`OPENEB_INSTALL_DIR`) with `CMAKE_INSTALL_PREFIX` variable
    and the directory where the Python packages will be deployed (`PYTHON3_PACKAGES_INSTALL_DIR`) using the `PYTHON3_SITE_PACKAGES` variable
    when generating the solution in step 2:

    ```bash
    cmake -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> -DCMAKE_INSTALL_PREFIX=<OPENEB_INSTALL_DIR> -DPYTHON3_SITE_PACKAGES=<PYTHON3_PACKAGES_INSTALL_DIR> -DBUILD_TESTING=OFF ..
    ```
    
  *  You can now launch the actual compilation and installation of the OpenEB files (your console should be launched as an administrator) :

    ```bash
    cmake --build . --config Release --parallel 4
    cmake --build . --config Release --target install
    ```
    
  * You also need to edit the `PATH`, `HDF5_PLUGIN_PATH` and `PYTHONPATH` environment variables:

    * append `<OPENEB_INSTALL_DIR>\bin` to the `PATH`
    * append `<OPENEB_INSTALL_DIR>\lib\hdf5\plugin` to the `HDF5_PLUGIN_PATH`
    * append `<PYTHON3_PACKAGES_INSTALL_DIR>` to the `PYTHONPATH`

  * If you did not customize the install folders when generating the solution, the `PYTHONPATH` environment variable needs
    not be modified and the `OPENEB_INSTALL_DIR` can be replaced by `C:\Program Files\Prophesee` in the previous instructions.


#### Compilation using MS Visual Studio

Open a command prompt inside the `openeb` folder (absolute path to this directory is called `OPENEB_SRC_DIR` in next sections) and do as follows:

 1. Create and open the build directory, where temporary files will be created: `mkdir build && cd build`
 2. Generate the Visual Studio files using CMake: `cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> ..` (adapt to your Visual Studio version).
    Note that the value passed to the parameter `-DCMAKE_TOOLCHAIN_FILE` must be an absolute path, not a relative one.
 3. Open the solution file `metavision.sln`, select the `Release` configuration and build the `ALL_BUILD` project.



Once the compilation is done, you can choose to work directly from the `build` folder
or you can deploy the OpenEB files (applications, samples, libraries etc.) in a directory of your choice.

* Option 1 - working from the `build` folder

  * To use OpenEB directly from the `build` folder,
  you need to update the environment variables as done in the script `utils\scripts\setup_env.bat`

* Option 2 - deploying OpenEB

  * To deploy OpenEB, you need to build the `INSTALL` project.
  By default, files will be deployed in `C:\Program Files\Prophesee`


#### Camera Plugins

Since OpenEB 3.0.0, **Prophesee camera plugins** are included in OpenEB, but you need to install the drivers
for the cameras to be available on Windows. To do so, follow this procedure:

1. download [wdi-simple.exe from our file server](https://files.prophesee.ai/share/dists/public/drivers/FeD45ki5/wdi-simple.exe)
2. execute the following commands in a Command Prompt launched as an administrator:

```bash
wdi-simple.exe -n "EVK" -m "Prophesee" -v 0x04b4 -p 0x00f4
wdi-simple.exe -n "EVK" -m "Prophesee" -v 0x03fd -p 0x5832 -i 00
wdi-simple.exe -n "EVK" -m "Prophesee" -v 0x04b4 -p 0x00f5
wdi-simple.exe -n "EVK" -m "Prophesee" -v 0x04b4 -p 0x00f3
```

If you are using a third-party camera, you need to follow the instructions provided by the camera vendor
to install the driver and the camera plugin. Make sure that you reference the location of the plugin in
the `MV_HAL_PLUGIN_PATH` environment variable.


#### Getting Started

To get started with OpenEB, you can download some [sample recordings](https://docs.prophesee.ai/stable/datasets.html) 
and visualize them with [metavision_viewer](https://docs.prophesee.ai/stable/metavision_sdk/modules/driver/samples/viewer.html)
or you can stream data from your Prophesee-compatible event-based camera.

*Note* that since OpenEB 3.0.0, Prophesee camera plugins are included in the OpenEB repository, so you don't need to perform
any extra step to install them. If you are using a third-party camera, you need to install the plugin provided
by the camera vendor and specify the location of the plugin using the `MV_HAL_PLUGIN_PATH` environment variable.

### Running the test suite (Optional)

Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

 * Download [the files](https://dataset.prophesee.ai/index.php/s/tiP0wl0r5aW5efL) necessary to run the tests.
   Click `Download` on the top right folder. Beware of the size of the obtained archive which weighs around 1.2 Gb.
   
 * Extract and put the content of this archive to `<OPENEB_SRC_DIR>/`. For instance, the correct path of sequence `gen31_timer.raw` should be `<OPENEB_SRC_DIR>/datasets/openeb/gen31_timer.raw`.

 * To run the test suite you need to reconfigure your build environment using CMake and to recompile


   * Compilation using CMake

    1. Regenerate the build using CMake (note that `-DCMAKE_TOOLCHAIN_FILE` must be absolute path, not a relative one)::

        ```
        cd <OPENEB_SRC_DIR>/build
        cmake -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> -DBUILD_TESTING=ON ..
        ```
    2. Compile: `cmake --build . --config Release --parallel 4`


   * Compilation using MS Visual Studio

    1. Generate the Visual Studio files using CMake (adapt the command to your Visual Studio version):

        `cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> -DBUILD_TESTING=ON ..`

        Note that the value passed to the parameter `-DCMAKE_TOOLCHAIN_FILE` must be an absolute path, not a relative one.

    2. Open the solution file `metavision.sln`, select the `Release` configuration and build the `ALL_BUILD` project.

 * Running the test suite is then simply `ctest -C Release`
