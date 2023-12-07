## Metavision SDK docker container


ARG UBUNTU_VERSION=22.04


FROM ubuntu:${UBUNTU_VERSION}

ARG OPENEB_VERSION=4.4.0
ARG PYTHON_VERSION=python3.10-dev

SHELL [ "/bin/bash", "-c" ]


ENV DEBIAN_FRONTEND "noninteractive"
ENV TZ "Europe/Berlin"

RUN apt update -y && apt upgrade -y && apt install -y apt-utils build-essential software-properties-common wget unzip curl git cmake ca-certificates vim autoconf rsync automake mesa-utils sudo parallel \
    libopencv-dev libboost-all-dev libusb-1.0-0-dev \
    libhdf5-dev hdf5-tools libglew-dev libglfw3-dev libcanberra-gtk-module libboost-program-options-dev libeigen3-dev ffmpeg \
    ${PYTHON_VERSION} python3-pip python3-tk \
    libprotobuf-dev protobuf-compiler && \
    apt clean

RUN python3 -m pip install pip --upgrade && \ 
    python3 -m pip install "opencv-python==4.5.5.64" "sk-video==1.1.10" "fire==0.4.0" "numpy==1.23.4" pandas scipy h5py jupyter jupyterlab matplotlib "ipywidgets==7.6.5"  pytest command_runner \ 
    "numba==0.56.3" "profilehooks==1.12.0" "pytorch_lightning==1.8.6" "tqdm==4.63.0" "kornia==0.6.8"  llvmlite  "torchmetrics==0.7.2" "seaborn==0.11.2" && \
    pip cache purge

# PyBind  
RUN cd / && wget https://github.com/pybind/pybind11/archive/v2.6.0.zip && \
            unzip v2.6.0.zip && \ 
            rm  v2.6.0.zip && \
            cd pybind11-2.6.0 && \
            mkdir build && cd build && \
            cmake .. -DPYBIND11_TEST=OFF && \
            cmake --build . -- -j `nproc` && \
            cmake --build . --target install 

RUN echo Version ${UBUNTU_VERSION} && sleep 10


# OpenEB
RUN cd / && git clone --depth 1 --branch ${OPENEB_VERSION}  https://github.com/prophesee-ai/openeb.git && \
            cd openeb && \
            mkdir build && cd build && \
            cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF && \
            cmake --build . --config Release -- -j `nproc` \
            && . utils/scripts/setup_env.sh && \
            cmake --build . --target install
    
