# OpenEB

OpenEB is the open source project associated with [Metavision SDK](https://docs.prophesee.ai/stable/index.html)

It enables anyone to get a better understanding of event-based vision, directly interact with events and build
their own applications or camera plugins. As a camera manufacturer, ensure your customers benefit from the most advanced
event-based software suite available by building your own plugin. As a creator, scientist, academic, join and contribute
to the fast-growing event-based vision community.

OpenEB is composed of the [Open modules of Metavision SDK](https://docs.prophesee.ai/stable/modules.html#chapter-modules-and-packaging-open):
* HAL: Hardware Abstraction Layer to operate any event-based vision device.
* Base: Foundations and common definitions of event-based applications.
* Core: Generic algorithms for visualization, event stream manipulation.
* Core ML: Generic functions for Machine Learning, event_to_video and video_to_event pipelines.
* Stream: High-level abstraction built on the top of HAL to easily interact with event-based cameras.
* UI: Viewer and display controllers for event-based data.

OpenEB also contains the source code of [Prophesee camera plugins](https://docs.prophesee.ai/stable/installation/camera_plugins.html),
enabling to stream data from our event-based cameras and to read recordings of event-based data.
The supported cameras are:
* EVK2 - HD
* EVK3 - VGA/320/HD
* EVK4 - HD

This document describes how to compile and install the OpenEB codebase.
For further information, refer to our [online documentation](https://docs.prophesee.ai/) where you will find
some [tutorials](https://docs.prophesee.ai/stable/tutorials/index.html) to get you started in C++ or Python,
some [samples](https://docs.prophesee.ai/stable/samples.html) to discover how to use
[our API](https://docs.prophesee.ai/stable/api.html) and a more detailed
[description of our modules and packaging](https://docs.prophesee.ai/stable/modules.html).


## Compiling on Linux

Compilation and execution were tested on platforms that meet the following requirements:

  * Linux: Ubuntu 22.04 or 24.04 64-bit
  * Architecture: amd64 (a.k.a. x64)
  * Graphic card with support of OpenGL 3.0 minimum

Compilation on other platforms (alternate Linux distributions, different versions of Ubuntu, ARM processor architecture etc.)
was not tested. For those platforms some adjustments to this guide or to the code itself may be required.


### Upgrading OpenEB

If you are upgrading OpenEB from a previous version, you should first read carefully the [Release Notes](https://docs.prophesee.ai/stable/release_notes.html)
as some changes may impact your usage of our SDK (e.g. [API](https://docs.prophesee.ai/stable/api.html) updates)
and cameras (e.g. [firmware update](https://support.prophesee.ai/portal/en/kb/articles/evk-firmware-versions) might be necessary).

Then, you need to clean your system from previously installed Prophesee software. If after a previous compilation, you chose to
deploy the Metavision files in your system path, then go to the `build` folder in the source code directory and
launch the following command to remove those files:

```bash
sudo make uninstall
```

In addition, make a global check in your system paths (`/usr/lib`, `/usr/local/lib`, `/usr/include`, `/usr/local/include`)
and in your environment variables (`PATH`, `PYTHONPATH` and `LD_LIBRARY_PATH`) to remove occurrences of Prophesee or Metavision files.


### Retrieving OpenEB source code

To retrieve OpenEB source code, you can just clone the [GitHub repository](https://github.com/prophesee-ai/openeb):

```bash
git clone https://github.com/prophesee-ai/openeb.git --branch 5.1.1
```

In the following sections, absolute path to this directory is called ``OPENEB_SRC_DIR``

If you choose to download an archive of OpenEB from GitHub rather than cloning the repository,
you need to ensure that you select a `Full.Source.Code.*` archive instead of using
the automatically generated `Source.Code.*` archives. This is because the latter do not include
a necessary submodule.


### Prerequisites

Install the following dependencies:

```bash
sudo apt update
sudo apt -y install apt-utils build-essential software-properties-common wget unzip curl git cmake
sudo apt -y install libopencv-dev libboost-all-dev libusb-1.0-0-dev libprotobuf-dev protobuf-compiler
sudo apt -y install libhdf5-dev hdf5-tools libglew-dev libglfw3-dev libcanberra-gtk-module ffmpeg 
```

Optionally, if you want to run the tests, you need to install Google Gtest and Gmock packages.
For more details, see [Google Test User Guide](https://google.github.io/googletest/):

```bash
sudo apt -y install libgtest-dev libgmock-dev
```

For the [Python API](https://docs.prophesee.ai/stable/api/python/index.html#chapter-api-python), you will need Python and some additional libraries.
We support Python 3.9 and 3.10 on Ubuntu 22.04 and Python 3.11 and 3.12 on Ubuntu 24.04.

We recommend using Python with [virtualenv](https://virtualenv.pypa.io/en/latest/) to avoid conflicts with other installed Python packages.
So, first install it along with some Python development tools:

```bash
sudo apt -y install python3.x-venv python3.x-dev
# where "x" is 9, 10, 11 or 12 depending on your Python version
```

Next, create a virtual environment and install the necessary dependencies:

```bash
python3 -m venv /tmp/prophesee/py3venv --system-site-packages
/tmp/prophesee/py3venv/bin/python -m pip install pip --upgrade
/tmp/prophesee/py3venv/bin/python -m pip install -r OPENEB_SRC_DIR/utils/python/requirements_openeb.txt
```

Note that when creating the virtual environment, it is necessary to use the `--system-site-packages` option to ensure that
the SDK packages installed in the system directories are accessible. However, this option also makes your local
user site-packages (typically found in `~/.local/lib/pythonX.Y/site-packages`) visible by default.
To prevent this and maintain a cleaner virtual environment, you can set the environment variable `PYTHONNOUSERSITE` to true.

Optionally, you can run the `activate` command (`source /tmp/prophesee/py3venv/bin/activate`) to modify your shell's environment variables,
setting the virtual environment's Python interpreter and scripts as the default for your current session.
This allows you to use simple commands like `python` without needing to specify the full path each time.

The Python bindings of the C++ API rely on the [pybind11](https://github.com/pybind) library, specifically version 2.11.0.

*Note* that pybind11 is required only if you want to use the Python bindings of the C++ API .
You can opt out of creating these bindings by passing the argument `-DCOMPILE_PYTHON3_BINDINGS=OFF` at step 3 during compilation (see below).
In that case, you will not need to install pybind11, but you won't be able to use our Python interface to the C++ API.

Unfortunately, there is no pre-compiled version of pybind11 available, so you need to install it manually:

```bash
wget https://github.com/pybind/pybind11/archive/v2.11.0.zip
unzip v2.11.0.zip
cd pybind11-2.11.0/
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


### Compilation

 1. Create and open the build directory `OPENEB_SRC_DIR`: `mkdir build && cd build`
 2. Generate the makefiles using CMake: `cmake .. -DBUILD_TESTING=OFF`.
    If you want to specify to cmake which version of Python to consider, you should use the option ``-DPython3_EXECUTABLE=<path_to_python_to_use>``.
    This is useful, for example, when you have a more recent version of Python than the ones we support installed on your system.
    In that case, cmake would select it and compilation might fail.
 3. Compile: `cmake --build . --config Release -- -j 4`

Once the compilation is finished, you have two options: you can choose to work directly from the `build` folder
or you can deploy the OpenEB files in the system path (`/usr/local/lib`, `/usr/local/include`...).

* Option 1 - working from `build` folder

  * To use OpenEB directly from the `build` folder, you need to update some environment variables using this script
    (which you may add to your `~/.bashrc` to make it permanent):

    ```bash
    source utils/scripts/setup_env.sh
    ```

  * [Prophesee camera plugins](https://docs.prophesee.ai/stable/installation/camera_plugins.html) are included in OpenEB,
    but you still need to copy the udev rules files in the system path and reload them so that your camera is detected with this command:

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

    Note that you can also deploy the OpenEB files (applications, samples, libraries etc.) in a directory of your choice by using
    the `CMAKE_INSTALL_PREFIX` variable (`-DCMAKE_INSTALL_PREFIX=<OPENEB_INSTALL_DIR>`) when generating the makefiles
    in step 2. Similarly, you can configure the directory where the Python packages will be deployed using the
    `PYTHON3_SITE_PACKAGES` variable (`-DPYTHON3_SITE_PACKAGES=<PYTHON3_PACKAGES_INSTALL_DIR>`).

  * you also need to update `LD_LIBRARY_PATH` and `HDF5_PLUGIN_PATH`
    (which you may add to your `~/.bashrc` to make it permanent):

    ```bash
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/local/hdf5/lib/plugin  # On Ubuntu 22.04
    export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/local/lib/hdf5/plugin  # On Ubuntu 24.04
    ```

Note that if you are using a third-party camera, you need to install the plugin provided
by the camera vendor and specify the location of the plugin using the `MV_HAL_PLUGIN_PATH` environment variable.

To get started with OpenEB, you can download some [sample recordings](https://docs.prophesee.ai/stable/datasets.html) 
and visualize them with [metavision_viewer](https://docs.prophesee.ai/stable/samples/modules/stream/viewer.html)
or you can stream data from your Prophesee-compatible event-based camera.

### Running the test suite (Optional)

Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

 * Download [the files](https://kdrive.infomaniak.com/app/share/975517/2aa2545c-6b12-4478-992b-df2acfb81b38) necessary to run the tests.
   Click `Download` on the top right folder. Beware of the size of the obtained archive which weighs around 1.5 Gb.

 * Extract and put the content of this archive to `<OPENEB_SRC_DIR>/datasets`. 
   For instance, the correct path of sequence `gen31_timer.raw` should be `<OPENEB_SRC_DIR>/datasets/openeb/gen31_timer.raw`.

 * Regenerate the makefiles with the test options enabled:

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
as some changes may impact your usage of our SDK (e.g. [API](https://docs.prophesee.ai/stable/api.html) updates)
and cameras (e.g. [firmware update](https://support.prophesee.ai/portal/en/kb/articles/evk-firmware-versions) might be necessary).

Then, if you have previously installed any Prophesee's software, you will need to uninstall it first.
Remove the folders where you installed Metavision artifacts (check both the `build` folder of the source code and
`C:\Program Files\Prophesee` which is the default install path of the deployment step).

### Retrieving OpenEB source code

To retrieve OpenEB source code, you can just clone the [GitHub repository](https://github.com/prophesee-ai/openeb):

```bash
git clone https://github.com/prophesee-ai/openeb.git --branch 5.1.1
```

In the following sections, absolute path to this directory is called ``OPENEB_SRC_DIR``

If you choose to download an archive of OpenEB from GitHub rather than cloning the repository,
you need to ensure that you select a `Full.Source.Code.*` archive instead of using
the automatically generated `Source.Code.*` archives. This is because the latter do not include
a necessary submodule.

### Prerequisites

Some steps of this procedure don't work on FAT32 and exFAT file system.
Hence, make sure that you are using a NTFS file system before going further.

You must enable the support for long paths:
 * Hit the Windows key, type gpedit.msc and press Enter
 * Navigate to Local Computer Policy > Computer Configuration > Administrative Templates > System > Filesystem
 * Double-click the "Enable Win32 long paths" option, select the "Enabled" option and click "OK"

To compile OpenEB, you will need to install some extra tools:

 * install [git](https://git-scm.com/download/win)
 * install [CMake 3.26](https://cmake.org/files/v3.26/cmake-3.26.6-windows-x86_64.msi)
 * install Microsoft C++ compiler (64-bit). You can choose one of the following solutions:
    * For building only, you can install MS Build Tools (free, part of Windows 10 SDK package)
    * Install Microsoft Visual C++ compiler (MSVC, 64-bit version) included in
      `Visual Studio 2022 - Fall 2023 LTSC (version 17.8) <https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#evergreen-bootstrappers>`_.
    * Select "C++ build tools", check Windows 10 SDK is checked, and add English Language Pack
    * For development, you can also download and run [Visual Studio Installer](https://visualstudio.microsoft.com/downloads/)
 * install [vcpkg](https://github.com/microsoft/vcpkg) that will be used for installing dependencies:
    * download and extract [vcpkg version 2024.04.26](https://github.com/microsoft/vcpkg/archive/refs/tags/2024.04.26.zip) in a folder that we will refer as `VCPKG_SRC_DIR`
    * `cd <VCPKG_SRC_DIR>`
    * `bootstrap-vcpkg.bat`
    * `vcpkg update`
    * copy the ``vcpkg-openeb.json`` file located in the OpenEB source code at ``utils/windows``
      into `VCPKG_SRC_DIR` and rename it to ``vcpkg.json``
  * install the libraries by running:
    * `vcpkg install --triplet x64-windows --x-install-root installed`
  * Finally, download and install [ffmpeg](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z) and add the `bin` directory to your PATH.

Note that if you're using vcpkg across multiple projects or versions of OpenEB, it’s beneficial to streamline
the number of vcpkg installations you manage. To achieve this, you'll need the specific versions of
the libraries required. You can find these versions by cross-referencing our `vcpkg.json` file with the
[official vcpkg repository](https://github.com/microsoft/vcpkg/tree/2024.04.26/versions),
but for your convenience, we’ve listed them below:

  * libusb: 1.0.27
  * boost: 1.78.0
  * opencv: 4.8.0
  * dirent: 1.24.0
  * gtest: 1.14.0
  * pybind11: 2.12.0
  * glew: 2.2.0
  * glfw3: 3.4.0
  * hdf5: 1.14.2
  * protobuf: 3.21.12

#### Installing Python and libraries

* Download "Windows x86-64 executable installer" for one of these Python versions:
  * [Python 3.9](https://www.python.org/downloads/release/python-3913/)
  * [Python 3.10](https://www.python.org/downloads/release/python-31011/)
  * [Python 3.11](https://www.python.org/downloads/release/python-3119/)
  * [Python 3.12](https://www.python.org/downloads/release/python-3125/)
* Add Python install and script directories in your `PATH` and make sure they are listed before
  the `WindowsApps` folder which contains a Python alias launching the Microsoft Store. So, if you installed
  Python 3.9 in the default path, your user `PATH` should contain those three lines in that order:
  
```bash
%USERPROFILE%\AppData\Local\Programs\Python\Python39
%USERPROFILE%\AppData\Local\Programs\Python\Python39\Scripts
%USERPROFILE%\AppData\Local\Microsoft\WindowsApps
````
We recommend using Python with [virtualenv](https://virtualenv.pypa.io/en/latest/) to avoid conflicts with other installed Python packages.

Create a virtual environment and install the necessary dependencies:

```bash
python -m venv C:\tmp\prophesee\py3venv --system-site-packages
C:\tmp\prophesee\py3venv\Scripts\python -m pip install pip --upgrade
C:\tmp\prophesee\py3venv\Scripts\python -m pip install -r OPENEB_SRC_DIR\utils\python\requirements_openeb.txt
```

When creating the virtual environment, it is necessary to use the `--system-site-packages` option to ensure that
the SDK packages installed in the system directories are accessible. However, this option also makes your local
user site-packages visible by default.
To prevent this and maintain a cleaner virtual environment, you can set the environment variable `PYTHONNOUSERSITE` to true.

Optionally, you can run the `activate` command (`C:\tmp\prophesee\py3venv\Scripts\activate`) to modify your shell's environment variables,
setting the virtual environment's Python interpreter and scripts as the default for your current session.
This allows you to use simple commands like `python` without needing to specify the full path each time.


#### Prerequisites for the ML module

To use Machine Learning features, you need to install some additional dependencies.

First, if you have some Nvidia hardware with GPUs, you can optionally install [CUDA (11.6 or 11.7)](https://developer.nvidia.com/cuda-downloads)
and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) to leverage them with pytorch and libtorch.


### Compilation

#### Compilation using CMake

Open a command prompt inside the `OPENEB_SRC_DIR` folder :

 1. Create and open the build directory, where temporary files will be created: `mkdir build && cd build`
 2. Generate the makefiles using CMake: `cmake .. -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR>`.
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

  * To deploy OpenEB in the default folder (`C:\Program Files\Prophesee`), execute this command 
    (your console should be launched as an administrator):

    ```bash 
    cmake --build . --config Release --target install
    ```

  * To deploy OpenEB in another folder, you should generate the solution again (step 2 above)
    with the additional variable `CMAKE_INSTALL_PREFIX` having the value of your target folder (`OPENEB_INSTALL_DIR`).

    Similarly, to specify where the Python packages will be deployed (``PYTHON3_PACKAGES_INSTALL_DIR``), you should use
    the `PYTHON3_SITE_PACKAGES` variable.

    Here is an example of a command customizing those two folders:
    
    ```bash
    cmake .. -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> -DCMAKE_INSTALL_PREFIX=<OPENEB_INSTALL_DIR> -DPYTHON3_SITE_PACKAGES=<PYTHON3_PACKAGES_INSTALL_DIR> -DBUILD_TESTING=OFF
    ```
    
    After this command, you should launch the actual compilation and installation of OpenEB 
    (your console should be launched as an administrator):

    ```bash
    cmake --build . --config Release --parallel 4
    cmake --build . --config Release --target install
    ```

  * You also need to manually edit some environment variables:

    * append `<OPENEB_INSTALL_DIR>\bin` to `PATH` (`C:\Program Files\Prophesee\bin` if you used default configuration)
    * append `<OPENEB_INSTALL_DIR>\lib\metavision\hal\plugins` to `MV_HAL_PLUGIN_PATH` (`C:\Program Files\Prophesee\lib\metavision\hal\plugins` if you used default configuration)
    * append `<OPENEB_INSTALL_DIR>\lib\hdf5\plugin` to `HDF5_PLUGIN_PATH` (`C:\Program Files\Prophesee\lib\hdf5\plugin` if you used default configuration)
    * append `<PYTHON3_PACKAGES_INSTALL_DIR>` to `PYTHONPATH` (not needed if you used default configuration)
    

#### Compilation using MS Visual Studio

Open a command prompt inside the `OPENEB_SRC_DIR` folder:

 1. Create and open the build directory, where temporary files will be created: `mkdir build && cd build`
 2. Generate the Visual Studio files using CMake: `cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR>` (adapt to your Visual Studio version).
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

  * You also need to manually edit some environment variables:

    * append `<OPENEB_INSTALL_DIR>\bin` to `PATH` (`C:\Program Files\Prophesee\bin` if you used default configuration)
    * append `<OPENEB_INSTALL_DIR>\lib\metavision\hal\plugins` to `MV_HAL_PLUGIN_PATH` (`C:\Program Files\Prophesee\lib\metavision\hal\plugins` if you used default configuration)
    * append `<OPENEB_INSTALL_DIR>\lib\hdf5\plugin` to `HDF5_PLUGIN_PATH` (`C:\Program Files\Prophesee\lib\hdf5\plugin` if you used default configuration)
    * append `<PYTHON3_PACKAGES_INSTALL_DIR>` to `PYTHONPATH` (not needed if you used default configuration)


#### Camera Plugins

Prophesee camera plugins are included in OpenEB, but you need to install the drivers
for the cameras to be available on Windows. To do so, follow this procedure:

1. download [wdi-simple.exe from our file server](https://kdrive.infomaniak.com/app/share/975517/4f59e852-af5e-4e00-90fc-f213aad20edd)
2. execute the following commands in a Command Prompt launched as an administrator:

```bash
wdi-simple.exe -n "EVK" -m "Prophesee" -v 0x04b4 -p 0x00f4
wdi-simple.exe -n "EVK" -m "Prophesee" -v 0x04b4 -p 0x00f5
wdi-simple.exe -n "EVK" -m "Prophesee" -v 0x04b4 -p 0x00f3
```

If you own an EVK2 or an RDK2, there are a few additional steps to complete that are detailed in our online documentation
in the [Camera Plugin section of the OpenEB install guide](https://docs.prophesee.ai/stable/installation/windows_openeb.html#camera-plugins).

If you are using a third-party camera, you need to follow the instructions provided by the camera vendor
to install the driver and the camera plugin. Make sure that you reference the location of the plugin in
the `MV_HAL_PLUGIN_PATH` environment variable.


#### Getting Started

To get started with OpenEB, you can download some [sample recordings](https://docs.prophesee.ai/stable/datasets.html) 
and visualize them with [metavision_viewer](https://docs.prophesee.ai/stable/samples/modules/stream/viewer.html)
or you can stream data from your Prophesee-compatible event-based camera.


### Running the test suite (Optional)

Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

 * Download [the files](https://kdrive.infomaniak.com/app/share/975517/2aa2545c-6b12-4478-992b-df2acfb81b38) necessary to run the tests.
   Click `Download` on the top right folder. Beware of the size of the obtained archive which weighs around 1.5 Gb.
   
 * Extract and put the content of this archive to `<OPENEB_SRC_DIR>/datasets`. 
   For instance, the correct path of sequence `gen31_timer.raw` should be `<OPENEB_SRC_DIR>/datasets/openeb/gen31_timer.raw`.

 * To run the test suite you need to reconfigure your build environment using CMake and to recompile


   * Compilation using CMake

    1. Regenerate the build using CMake (note that `-DCMAKE_TOOLCHAIN_FILE` must be absolute path, not a relative one)::

        ```
        cd <OPENEB_SRC_DIR>/build
        cmake .. -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> -DBUILD_TESTING=ON
        ```
    2. Compile: `cmake --build . --config Release --parallel 4`


   * Compilation using MS Visual Studio

    1. Generate the Visual Studio files using CMake (adapt the command to your Visual Studio version and note that `-DCMAKE_TOOLCHAIN_FILE` must be absolute path, not a relative one):

        `cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=<OPENEB_SRC_DIR>\cmake\toolchains\vcpkg.cmake -DVCPKG_DIRECTORY=<VCPKG_SRC_DIR> -DBUILD_TESTING=ON`

    2. Open the solution file `metavision.sln`, select the `Release` configuration and build the `ALL_BUILD` project.

 * Running the test suite is then simply `ctest -C Release`
