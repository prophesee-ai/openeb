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

#include <cstdint>
#include <cstring>
#include "boards/v4l2/v4l2_device.h"

// Linux specific headers
#include <fcntl.h>
#include <linux/videodev2.h>
#include <string>
#include <sys/stat.h>
#include <sys/ioctl.h>

using namespace Metavision;

void Metavision::raise_error(const std::string &str) {
    throw std::runtime_error(str + " (" + std::to_string(errno) + " - " + std::strerror(errno) + ")");
}

V4L2DeviceControl::V4L2DeviceControl(const std::string &dev_name) {
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

V4l2RequestBuffers V4L2DeviceControl::request_buffers(v4l2_memory memory, uint32_t nb_buffers) {
    V4l2RequestBuffers req{0};
    req.count  = nb_buffers;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = memory;

    if (-1 == ioctl(fd_, VIDIOC_REQBUFS, &req)) {
        raise_error("VIDIOC_QUERYBUF failed");
    }

    return req;
}

V4l2Buffer V4L2DeviceControl::query_buffer(v4l2_memory memory_type, uint32_t buf_index) {
    V4l2Buffer buf{0};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = memory_type;
    buf.index  = buf_index;

    if (ioctl(fd_, VIDIOC_QUERYBUF, &buf))
        raise_error("VIDIOC_QUERYBUF failed");

    return buf;
}

V4l2Capability V4L2DeviceControl::get_capability() const {
    return cap_;
}

int V4L2DeviceControl::queue_buffer(V4l2Buffer &buffer) {
    auto ioctl_res = ioctl(fd_, VIDIOC_QBUF, &buffer);
    if (ioctl_res) {
        raise_error("VIDIOC_QBUF failed");
    }
    return ioctl_res;
}

int V4L2DeviceControl::dequeue_buffer(V4l2Buffer *buffer) {
    return ioctl(fd_, VIDIOC_DQBUF, buffer);
}

void V4L2DeviceControl::start() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type))
        raise_error("VIDIOC_STREAMON failed");
}
void V4L2DeviceControl::stop() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMOFF, &type))
        raise_error("VIDIOC_STREAMOFF failed");
}
void V4L2DeviceControl::reset() {}
