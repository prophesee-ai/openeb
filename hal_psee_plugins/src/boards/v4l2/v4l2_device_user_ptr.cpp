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

#include "boards/v4l2/dma_buf_heap.h"
#include "boards/v4l2/v4l2_device_user_ptr.h"

namespace Metavision {

void V4l2DeviceUserPtr::allocate_buffers(unsigned int nb_buffers) {
    for (unsigned int i = 0; i < nb_buffers; ++i) {
        /* Get a buffer using CMA allocator in user space. */
        auto dmabuf_fd = dma_buf_heap_->alloc(length_);

        void *start = mmap(NULL, length_, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
        if (MAP_FAILED == start)
            raise_error("mmap failed");

        dma_buf_heap_->cpu_sync_start(dmabuf_fd);
        memset(start, 0, length_);

        std::cout << "Allocate buffer: " << i << " at: " << std::hex << start << " of " << std::dec << length_
                  << " bytes." << std::endl;

        /* Record the handle to manage the life cycle. */
        buffers_desc_.push_back(BufferDesc{start, dmabuf_fd});
    }
}

void V4l2DeviceUserPtr::free_buffers() {
    int i = get_nb_buffers();

    while (0 < i) {
        auto idx = get_buffer();
        std::cout << "Release " << i << " buffer: " << idx << std::endl;
        auto buf = buffers_desc_.at(idx);
        if (-1 == munmap(buf.start, length_))
            raise_error("munmap failed");
        dma_buf_heap_->free(buf.dmabuf_fd);
        --i;
    }

    buffers_desc_.clear();
}

unsigned int V4l2DeviceUserPtr::get_nb_buffers() const {
    return buffers_desc_.size();
}

V4l2DeviceUserPtr::V4l2DeviceUserPtr(const std::string &dev_name, std::unique_ptr<Metavision::DmaBufHeap> dma_buf_heap,
                                     std::size_t length, unsigned int nb_buffers) :
    V4l2Device(dev_name), dma_buf_heap_(std::move(dma_buf_heap)), length_(length) {
    auto granted_buffers = request_buffers(V4L2_MEMORY_USERPTR, nb_buffers);
    std::cout << "Requested buffers: " << nb_buffers << " granted buffers: " << granted_buffers << std::endl;
    allocate_buffers(granted_buffers);
}

V4l2DeviceUserPtr::~V4l2DeviceUserPtr() {
    free_buffers();
}

/** Release the buffer designed by the index to the driver. */
void V4l2DeviceUserPtr::release_buffer(int idx) const {
    auto desc = buffers_desc_.at(idx);

    dma_buf_heap_->cpu_sync_stop(desc.dmabuf_fd);
    std::cout << "Release buffer: " << idx << " at " << std::hex << desc.start << " of " << std::dec << length_
              << " bytes." << std::endl;
    struct v4l2_buffer buf;
    std::memset(&buf, 0, sizeof(buf));
    buf.type      = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory    = V4L2_MEMORY_USERPTR;
    buf.index     = idx;
    buf.m.userptr = (unsigned long)desc.start;
    buf.length    = length_;
    if (ioctl(fd_, VIDIOC_QBUF, &buf))
        raise_error("VIDIOC_QBUF failed");
}

/** Poll a MIPI frame buffer through the V4L2 interface.
 * Return the buffer index.
 * */
int V4l2DeviceUserPtr::get_buffer() const {
    struct v4l2_buffer buf;
    std::memset(&buf, 0, sizeof(buf));
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_USERPTR;
    while (ioctl(fd_, VIDIOC_DQBUF, &buf)) {}

    int idx   = buf.index;
    auto desc = buffers_desc_.at(idx);
    dma_buf_heap_->cpu_sync_start(desc.dmabuf_fd);
    return idx;
}

/** Return the buffer address and size (in bytes) designed by the index. */
std::pair<void *, std::size_t> V4l2DeviceUserPtr::get_buffer_desc(int idx) const {
    auto desc = buffers_desc_.at(idx);
    return std::make_pair(desc.start, nb_not_null_data(desc.start, length_));
}

} // namespace Metavision
