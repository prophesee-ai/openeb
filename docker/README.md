Docker environment for OpenEB
=============================


This is a basic dockerized instalation of the OpenEB SDK. Currently only amd64 architecture is supported and Ubuntu 20.04 and 22.04.

The Dockerfile includes all the Nvidia dependencies allowing the image to be used with GPU enabled machines (using the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)). Nevertheless, this is only useful for accelerating certian ML examples. 

## Building
```bash

# Build latest OpenEB on Ubuntu 22.04 and Python 3.10
docker build -t openeb:4.4.0 -f Dockerfile .

# Additional build arguments
docker build --build-args OPENEB_VERSION=4.3.0 --build-args UBUNTU_VERSION=20.04 --build-args PYTHON_VERSION=python3.8-dev
```

## Usage

```bash

# Running Metavision Viewer with access to USB camera
xhost '+local:*';

docker run -it --privileged --gpus all -v /dev/bus/usb:/deb/bus/usb -e DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix/ --rm --net=host openeb:4.4.0 metavision_viewer
```
