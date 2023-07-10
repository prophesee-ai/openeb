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
#include <memory>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "boards/v4l2/v4l2_device_mmap.h"

using namespace Metavision;

V4l2DeviceMmap::V4l2DeviceMmap(std::shared_ptr<V4l2Device> device, unsigned int nb_buffers) : fd_(device->get_fd()) {
    auto granted_buffers = device->request_buffers(V4L2_MEMORY_MMAP, nb_buffers);
    query_buffers(granted_buffers);
}

V4l2DeviceMmap::~V4l2DeviceMmap() {
    free_buffers();
}
/** Release the buffer designed by the index to the driver. */
void V4l2DeviceMmap::release_buffer(int idx) const {
    struct v4l2_buffer buf;
    std::memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = idx;
    if (ioctl(fd_, VIDIOC_QBUF, &buf))
        raise_error("VIDIOC_QBUF failed");
}

/** Poll a MIPI frame buffer through the V4L2 interface.
 * Return the buffer index.
 * */
int V4l2DeviceMmap::get_buffer() const {
    struct v4l2_buffer buf;
    std::memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    while (ioctl(fd_, VIDIOC_DQBUF, &buf)) {}

    return buf.index;
}

/** Return the buffer address and size (in bytes) designed by the index. */
std::pair<void *, std::size_t> V4l2DeviceMmap::get_buffer_desc(int idx) const {
    auto desc = buffers_desc_.at(idx);
    return std::make_pair(desc.start, V4l2Device::nb_not_null_data(desc.start, desc.length));
}

void V4l2DeviceMmap::query_buffers(unsigned int nb_buffers) {
    for (unsigned int i = 0; i < nb_buffers; ++i) {
        /* Get a buffer allocated in Kernel space. */
        struct v4l2_buffer buf;
        std::memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf))
            raise_error("VIDIOC_QUERYBUF failed");

        void *start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
        if (MAP_FAILED == start)
            raise_error("mmap failed");
        memset(start, 0, buf.length);

        /* Record the handle to manage the life cycle. */
        buffers_desc_.push_back(BufferDesc{start, buf.length});
    }
}

void V4l2DeviceMmap::free_buffers() {
    /* Close dmabuf fd */
    for (const auto &buf : buffers_desc_) {
        if (-1 == munmap(buf.start, buf.length))
            raise_error("munmap failed");
    }
    buffers_desc_.clear();
}

unsigned int V4l2DeviceMmap::get_nb_buffers() const {
    return buffers_desc_.size();
}
