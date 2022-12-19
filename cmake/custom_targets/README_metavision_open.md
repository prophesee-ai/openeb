# OpenEB

OpenEB is the open source project associated with [Metavision Intelligence](https://www.prophesee.ai/metavision-intelligence/)

It enables anyone to get a better understanding of event-based vision, directly interact with events and build
their own applications or plugins. As a camera manufacturer, ensure your customers benefit from the most advanced 
event-based software suite available by building your own plugin. As a creator, scientist, academic, join and contribute
to the fast-growing event-based vision community.

OpenEB is composed of the Open modules of Metavision Intelligence:
* HAL: Hardware Abstraction Layer to operate any event-based vision device.
* Base: Foundations and common definitions of event-based applications.
* Core: Generic algorithms for visualization, event stream manipulation, applicative pipeline generation.
* Core ML: Generic functions for Machine Learning, event_to_video and video_to_event pipelines.
* Driver: High-level abstraction built on the top of HAL to easily interact with event-based cameras.
* UI: Viewer and display controllers for event-based data.

OpenEB also contains the source code of Prophesee camera plugins, enabling to stream data from our event-based cameras
and to read recordings of event-based data. The supported cameras are:
* EVK1 - Gen3/Gen3.1 VGA
* EVK2 - Gen4.1 HD
* EVK3 - Gen 3.1 VGA / Gen4.1 HD
* EVK4 - HD

This document describes how to compile and install the OpenEB codebase.
For further information, refer to our [online documentation](https://docs.prophesee.ai/) where you will find
some [tutorials](https://docs.prophesee.ai/stable/metavision_sdk/tutorials/index.html) to get you started in C++ or Python,
some [samples](https://docs.prophesee.ai/stable/samples.html) to discover how to use
[our API](https://docs.prophesee.ai/stable/api.html) and a more detailed
[description of our modules and packaging](https://docs.prophesee.ai/stable/modules.html).


## Compiling on Ubuntu

Currently, we support Ubuntu 18.04 and 20.04. 
Compilation on other versions of Ubuntu or other Linux distributions was not tested.
For those platforms some adjustments to this guide or to the code itself may be required (specially for non-Debian Linux).

### Upgrading OpenEB

If you are upgrading OpenEB from a previous version, you should first read carefully the `Release Notes <https://docs.prophesee.ai/stable/release_notes.html>`_
as some changes may impact your usage of our SDK (e.g. API updates) and cameras (e.g. `firmware update <https://support.prophesee.ai/portal/en/kb/articles/evk-firmware-versions>`_ might be necessary).

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
sudo apt -y install libopencv-dev googletest libgtest-dev libboost-all-dev libusb-1.0-0-dev
sudo apt -y install libglew-dev libglfw3-dev libcanberra-gtk-module ffmpeg
```

Optionally, if you want to run the tests, you need to compile the `GoogleTest <https://google.github.io/googletest/>`_ package:

```bash
cd /usr/src/googletest
sudo cmake .
sudo make
sudo make install
```

For the Python API, you will need Python and some additional libraries.
If Python is not available on your system, install it
(we support Python 3.6 and 3.7 on Ubuntu 18.04 and Python 3.7 and 3.8 on Ubuntu 20.04).

Then install `pip` and some Python libraries:
```bash
sudo apt -y install python3-pip python3-distutils
sudo apt -y install python3.X-dev  # where X is 6, 7 or 8 depending on your Python version (3.6, 3.7 or 3.8)
python3 -m pip install pip --upgrade
python3 -m pip install "opencv-python>=4.5.5.64" "sk-video==1.1.10" "fire==0.4.0" "numpy<=1.21" pandas scipy h5py 
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

First, if you have some Nvidia hardware with GPUs, you can optionally install `CUDA (10.2 or 11.1) <https://developer.nvidia.com/cuda-downloads>`_
and `cuDNN <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>`_ to leverage them with pytorch and libtorch.

Make sure that you install a version of CUDA that is compatible with your GPUs by checking
`Nvidia compatibility page <https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html>`_.

Note that, at the moment, we don't support `OpenCL <https://www.khronos.org/opencl/>`_ and AMD GPUs.

Then, install PyTorch 1.8.2 LTS. This version was deprecated by PyTorch team but can still be downloaded
in `the Previous Versions page of pytorch.org <https://pytorch.org/get-started/previous-versions/#v182-with-lts-support>`_
(in future releases of Metavision ML, more recent version of PyTorch will be leveraged).
Retrieve and execute the pip command for the installation. Here is an example of a command that can be retrieved for
pytorch using CUDA 11.1:

```bash
python3 -m pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```

Then install some extra Python libraries:

```bash
python3 -m pip install numba profilehooks "pytorch_lightning==1.5.10" "tqdm==4.63.0" "kornia==0.6.1"
```

### Compilation

 1. Retrieve the code `git clone https://github.com/prophesee-ai/openeb.git`
 2. Create and open the build directory in the `openeb` folder (absolute path to this directory is called `OPENEB_SRC_DIR` in next sections): `cd openeb; mkdir build && cd build`
 3. Generate the makefiles using CMake: `cmake .. -DBUILD_TESTING=OFF`
 4. Compile: `cmake --build . --config Release -- -j 4`
 
To use OpenEB directly from the build folder, update your environment variables using this script
(which you may add to your ~/.bashrc to make it permanent):

```bash
source <OPENEB_SRC_DIR>/build/utils/scripts/setup_env.sh
```

Optionally, you can deploy the OpenEB files in the system paths to use them as 3rd party dependency in some other code
with the following command: `sudo cmake --build . --target install`.

In that case, you will also need to update:

  * `LD_LIBRARY_PATH` with `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib`

If you want those settings to be permanent, you should add the previous commands in your ~/.bashrc.

You can also deploy the OpenEB files (applications, samples, libraries etc.) in a directory of your choice by using 
the `CMAKE_INSTALL_PREFIX` variable (`-DCMAKE_INSTALL_PREFIX=<OPENEB_INSTALL_DIR>`) when generating the makefiles
in step 3. Similarly, you can configure the directory where the Python packages will be deployed using the
`PYTHON3_SITE_PACKAGES` variable (`-DPYTHON3_SITE_PACKAGES=<PYTHON3_PACKAGES_INSTALL_DIR>`).

Since OpenEB 3.0.0, Prophesee camera plugins are included in OpenEB. If you did not perform the optional deployment step
(`sudo cmake --build . --target install`) and instead used “setup_env.sh”, then you need to copy the udev rules files 
used by Prophesee cameras in the system path and reload them so that your camera is detected with this command:

```bash
sudo cp <OPENEB_SRC_DIR>/hal_psee_plugins/resources/rules/*.rules /etc/udev/rules.d
sudo udevadm control --reload-rules
sudo udevadm trigger
```

If you are using a third-party camera, you need to install the plugin provided by the camera vendor and specify
the location of the plugin using the `MV_HAL_PLUGIN_PATH` environment variable.

To get started with OpenEB, you can download some [sample recordings](https://docs.prophesee.ai/stable/datasets.html) 
and visualize them with [metavision_viewer](https://docs.prophesee.ai/stable/metavision_sdk/modules/driver/guides/viewer.html#chapter-sdk-driver-samples-viewer)
or you can stream data from your Prophesee-compatible event-based camera.

### Running the test suite (Optional)

Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

 * Download [the files](https://dataset.prophesee.ai/index.php/s/hyCzGM4tpR8w5bx) necessary to run the tests.
   Click `Download` on the top right folder. Beware of the size of the obtained archive which weighs around 500 Mb.

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

If you are upgrading OpenEB from a previous version, you should first read carefully the `Release Notes <https://docs.prophesee.ai/stable/release_notes.html>`_
as some changes may impact your usage of our SDK (e.g. :API updates) and cameras (e.g. `firmware update <https://support.prophesee.ai/portal/en/kb/articles/evk-firmware-versions>`_ might be necessary).

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
 * install [CMake 3.20](https://cmake.org/files/v3.20/cmake-3.20.6-windows-x86_64.msi)
 * install Microsoft C++ compiler (64-bit). You can choose one of the following solutions:
    * For building only, you can install MS Build Tools (free, part of Windows 10 SDK package)
      * Download and run ["Build tools for Visual Studio 2019" installer](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
      * Select "C++ build tools", make sure Windows 10 SDK is checked, and add English Language Pack
    * For development, you can also download and run [Visual Studio Installer](https://visualstudio.microsoft.com/fr/downloads/)    
 * install [vcpkg](https://github.com/microsoft/vcpkg) that will be used for installing dependencies:
    * download and extract [vcpkg version 2022.03.10](https://github.com/microsoft/vcpkg/archive/refs/tags/2022.03.10.zip)
    * `cd <VCPKG_SRC_DIR>`
    * `bootstrap-vcpkg.bat`
  * install the libraries by running `vcpkg.exe install --triplet x64-windows libusb eigen3 boost opencv glfw3 glew gtest dirent`
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

#### Installing Python and libraries

* Download "Windows x86-64 executable installer" for one of these Python versions:
  * [Python 3.7](https://www.python.org/downloads/release/python-379/)
  * [Python 3.8](https://www.python.org/downloads/release/python-3810/)
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
python -m pip install "opencv-python>=4.5.5.64" "sk-video==1.1.10" "fire==0.4.0" "numpy<=1.21" pandas scipy h5py
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

First, if you have some Nvidia hardware with GPUs, you can optionally install `CUDA (10.2 or 11.1) <https://developer.nvidia.com/cuda-downloads>`_
and `cuDNN <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>`_ to leverage them with pytorch and libtorch.

Then, install pytorch. Go to `pytorch.org <https://pytorch.org>`_ to retrieve the pip command that you
will launch in a console to install PyTorch 1.8.2 LTS. Here is an example of a command that can be retrieved for
pytorch using CUDA 11.1:

```bash
python -m pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

Then install some extra Python libraries:

```bash
python -m pip install numba profilehooks "pytorch_lightning==1.5.10" "tqdm==4.63.0" "kornia==0.6.1"
```

### Compilation

First, retrieve the codebase:

```bash
git clone https://github.com/prophesee-ai/openeb.git
```

#### Compilation using CMake

Open a command prompt inside the `openeb` folder (absolute path to this directory is called `OPENEB_SRC_DIR` in next sections) and do as follows:

 1. Create and open the build directory, where temporary files will be created: `mkdir build && cd build`
 2. Generate the makefiles using CMake: `cmake -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> ..`.
    Note that the value passed to the parameter `-DCMAKE_TOOLCHAIN_FILE` must be an absolute path, not a relative one. 
 3. Compile: `cmake --build . --config Release --parallel 4`
 
To use OpenEB directly from the build folder, update your environment variables using this script:

```bash
<OPENEB_SRC_DIR>\build\utils\scripts\setup_env.bat
```

Optionally, you can deploy the OpenEB files (applications, samples, libraries etc.) in a directory of your choice.
To do so, configure the target folder (`OPENEB_INSTALL_DIR`) with `CMAKE_INSTALL_PREFIX` variable 
(default value is `C:\Program Files\Prophesee`) when generating the makefiles in step 2:

```bash
cmake .. -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> -DCMAKE_INSTALL_PREFIX=<OPENEB_INSTALL_DIR> -DBUILD_TESTING=OFF
```

You can also configure the directory where the Python packages will be deployed using the `PYTHON3_SITE_PACKAGES` variable
(note that in that case, you will also need to edit your environment variable `PYTHONPATH` and append the `<PYTHON3_PACKAGES_INSTALL_DIR>` path):

```bash
cmake .. -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> -DCMAKE_INSTALL_PREFIX=<OPENEB_INSTALL_DIR> -DPYTHON3_SITE_PACKAGES=<PYTHON3_PACKAGES_INSTALL_DIR> -DBUILD_TESTING=OFF
```

Once you performed this configuration, you can launch the actual installation of the OpenEB files:

```bash
cmake --build . --config Release --target install
```

#### Compilation using MS Visual Studio

Open a command prompt inside the `openeb` folder (absolute path to this directory is called `OPENEB_SRC_DIR` in next sections) and do as follows:

 1. Create and open the build directory, where temporary files will be created: `mkdir build && cd build`
 2. Generate the Visual Studio files using CMake: `cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> ..` (adapt to your Visual Studio version).
    Note that the value passed to the parameter `-DCMAKE_TOOLCHAIN_FILE` must be an absolute path, not a relative one.
 3. Open the solution file `metavision.sln`, select the `Release` configuration and build the `ALL_BUILD` project.

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
and visualize them with [metavision_viewer](https://docs.prophesee.ai/stable/metavision_sdk/modules/driver/guides/viewer.html#chapter-sdk-driver-samples-viewer)
or you can stream data from your Prophesee-compatible event-based camera.

*Note* that since OpenEB 3.0.0, Prophesee camera plugins are included in the OpenEB repository, so you don't need to perform
any extra step to install them. If you are using a third-party camera, you need to install the plugin provided
by the camera vendor and specify the location of the plugin using the `MV_HAL_PLUGIN_PATH` environment variable.

### Running the test suite (Optional)

Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

 * Download [the files](https://dataset.prophesee.ai/index.php/s/hyCzGM4tpR8w5bx) necessary to run the tests.
   Click `Download` on the top right folder. Beware of the size of the obtained archive which weighs around 500 Mb.
   
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

        `cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> -DBUILD_TESTING=ON ..`

        Note that the value passed to the parameter `-DCMAKE_TOOLCHAIN_FILE` must be an absolute path, not a relative one.

    2. Open the solution file `metavision.sln`, select the `Release` configuration and build the `ALL_BUILD` project.

 * Running the test suite is then simply `ctest -C Release`
