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

#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

#include <sys/ioctl.h>
#include <sys/mman.h>

#include "boards/v4l2/v4l2_device.h"
#include "boards/v4l2/v4l2_device_mmap.h"

using namespace Metavision;

V4l2DeviceMmap::V4l2DeviceMmap(std::shared_ptr<V4l2Device> device, unsigned int nb_buffers) : device_(device) {
    auto granted_buffers = device->request_buffers(V4L2_MEMORY_MMAP, nb_buffers);

    for (uint32_t i = 0; i < granted_buffers.count; ++i) {
        auto buf = device->query_buffer(V4L2_MEMORY_MMAP, i);

        void *start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, device->get_fd(), buf.m.offset);
        if (MAP_FAILED == start)
            raise_error("mmap failed");

        memset(start, 0, buf.length);

        /* Record the handle to manage the life cycle. */
        buffers_desc_.push_back(BufferDesc{start, buf.length});
    }
}

V4l2DeviceMmap::~V4l2DeviceMmap() {
    free_buffers();
}

/** Release the buffer designed by the index to the driver. */
void V4l2DeviceMmap::release_buffer(int idx) const {
    V4l2Buffer buf;
    std::memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index  = idx;
    device_->queue_buffer(buf);
}

/** Poll a MIPI frame buffer through the V4L2 interface.
 * Return the buffer index.
 * */
int V4l2DeviceMmap::poll_buffer() const {
    V4l2Buffer buf{0};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    while (device_->dequeue_buffer(&buf)) {
        using namespace std::literals::chrono_literals;
        std::this_thread::sleep_for(1ms);
    }

    return buf.index;
}

/** Return the buffer address and size (in bytes) designed by the index. */
std::pair<void *, std::size_t> V4l2DeviceMmap::get_buffer_desc(int idx) const {
    auto desc = buffers_desc_.at(idx);
    return std::make_pair(desc.start, V4l2Device::nb_not_null_data(desc.start, desc.length));
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
