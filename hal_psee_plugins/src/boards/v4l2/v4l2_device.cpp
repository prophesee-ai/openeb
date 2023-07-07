/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#include <cstring>
#include "boards/v4l2/v4l2_device.h"
#include "metavision/hal/utils/detail/hal_log_impl.h"

// Linux specific headers
#include <linux/videodev2.h>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <sys/ioctl.h>

using namespace Metavision;

void Metavision::raise_error(const std::string &str) {
    throw std::runtime_error(str + " (" + std::to_string(errno) + " - " + std::strerror(errno) + ")");
}

V4l2Device::V4l2Device(const std::string &dev_name) {
    struct stat st;
    if (-1 == stat(dev_name.c_str(), &st))
        raise_error(dev_name + "Cannot identify device.");

    if (!S_ISCHR(st.st_mode))
        throw std::runtime_error(dev_name + " is not a device");

    fd_ = open(dev_name.c_str(), O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == fd_) {
        raise_error(dev_name + "Cannot open device");
    }

    if (ioctl(fd_, VIDIOC_QUERYCAP, &cap_)) {
        if (EINVAL == errno) {
            throw std::runtime_error(dev_name + " is not a V4L2 device");
        } else {
            raise_error("VIDIOC_QUERYCAP failed");
        }
    }
    MV_HAL_LOG_INFO() << "V4l2 Device opened : \n"
                      << "  - card : " << cap_.card << "\n"
                      << "  - driver : " << cap_.driver << "\n"
                      << "  - bus info : " << cap_.bus_info << "\n"
                      << "  - capabilities : " << cap_.capabilities << "\n"
                      << "  - device cap : " << cap_.device_caps << "\n";

    if (!(cap_.capabilities & V4L2_CAP_VIDEO_CAPTURE))
        throw std::runtime_error(dev_name + " is not video capture device");
    if (!(cap_.capabilities & V4L2_CAP_STREAMING))
        throw std::runtime_error(dev_name + " does not support streaming i/o");

    struct v4l2_format fmt;
    std::memset(&fmt, 0, sizeof(fmt));
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;
    fmt.fmt.pix.width       = 65536;
    fmt.fmt.pix.height      = 64;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt))
        raise_error("VIDIOC_S_FMT failed");
}

unsigned int V4l2Device::request_buffers(v4l2_memory memory, unsigned int nb_buffers) {
    struct v4l2_requestbuffers req;
    std::memset(&req, 0, sizeof(req));
    req.count  = nb_buffers;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = memory;

    if (-1 == ioctl(fd_, VIDIOC_REQBUFS, &req)) {
        raise_error("VIDIOC_QUERYBUF failed");
    }
    return req.count;
}

void V4l2Device::start() {
    MV_HAL_LOG_TRACE() << " Nb buffers pre allocated: " << get_nb_buffers() << std::endl;
    for (unsigned int i = 0; i < get_nb_buffers(); ++i) {
        release_buffer(i);
    }
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type))
        raise_error("VIDIOC_STREAMON failed");
}

void V4l2Device::stop() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMOFF, &type))
        raise_error("VIDIOC_STREAMOFF failed");
}
