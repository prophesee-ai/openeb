# OpenEB Windows Build, Test, and Installation Guide

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
    * download and extract [vcpkg version 2023.11.20](https://github.com/microsoft/vcpkg/archive/refs/tags/2023.11.20.zip)
    * `cd <VCPKG_SRC_DIR>`
    * `bootstrap-vcpkg.bat`
  * install the libraries by running `vcpkg.exe install --triplet x64-windows libusb boost opencv dirent gtest glew glfw3 hdf5[cpp,threadsafe,tools,zlib]`
  * Finally, download and install [ffmpeg](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full.7z) and add the `bin` directory to your PATH.

Note that if you are using vcpkg for various projects or multiple versions of OpenEB, you might want to optimize the
number of vcpkg install you manage. To do so, you will need the versions of the libraries we require.
Those can be found in the [vcpkg repository](https://github.com/microsoft/vcpkg/tree/2023.11.20/versions) but we list them here for convenience:
  * libusb: 1.0.26
  * boost: 1.83.0
  * opencv: 4.8.0
  * dirent: 1.24.0
  * gtest: 1.14.0
  * glew: 2.2.0
  * glfw3: 3.3.8
  * hdf5: 1.14.2

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
python -m pip install "opencv-python==4.5.5.64" "sk-video==1.1.10" "fire==0.4.0" "numpy==1.23.4" "h5py==3.7.0" pandas scipy
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

Then, you need to install [PyTorch 1.13.1](https://pytorch.org).
Retrieve and execute the pip command of version 1.13.1 from
the [previous versions install guide section](<https://pytorch.org/get-started/previous-versions/#v1131>).

Then install some extra Python libraries:

```bash
python -m pip install "numba==0.56.3" "profilehooks==1.12.0" "pytorch_lightning==1.8.6" "tqdm==4.63.0" "kornia==0.6.8"
```

### Compilation

First, retrieve the codebase:

```bash
git clone https://github.com/prophesee-ai/openeb.git --branch 4.5.2
```

Note that if you choose to download an archive of OpenEB from GitHub rather than cloning the repository,
you need to ensure that you select a ``Full.Source.Code.*`` archive instead of using
the automatically generated ``Source.Code.*`` archives. This is because the latter do not include
a necessary submodule.


#### Compilation using CMake

Open a command prompt inside the `openeb` folder (absolute path to this directory is called `OPENEB_SRC_DIR` in next sections) and do as follows:

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

  * To deploy SDK Pro in the default folder (`C:\Program Files\Prophesee`), execute this command
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

Open a command prompt inside the `openeb` folder (absolute path to this directory is called `OPENEB_SRC_DIR` in next sections) and do as follows:

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


#### Camera Plugins

Prophesee camera plugins are included in OpenEB, but you need to install the drivers
for the cameras to be available on Windows. To do so, follow this procedure:

1. download [wdi-simple.exe from our file server](https://files.prophesee.ai/share/dists/public/drivers/FeD45ki5/wdi-simple.exe)
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
and visualize them with [metavision_viewer](https://docs.prophesee.ai/stable/samples/modules/driver/viewer.html)
or you can stream data from your Prophesee-compatible event-based camera.


### Running the test suite (Optional)

Running the test suite is a sure-fire way to ensure you did everything well with your compilation and installation process.

 * Download [the files](https://dataset.prophesee.ai/index.php/s/tiP0wl0r5aW5efL) necessary to run the tests.
   Click `Download` on the top right folder. Beware of the size of the obtained archive which weighs around 1.2 Gb.

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
