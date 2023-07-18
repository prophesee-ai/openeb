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

#include "boards/v4l2/v4l2_device.h"
#include "boards/v4l2/v4l2_data_transfer.h"

#include "metavision/hal/utils/hal_log.h"

using namespace Metavision;

V4l2DataTransfer::V4l2DataTransfer(std::shared_ptr<V4l2Device> device, uint32_t raw_event_size_bytes) :
    DataTransfer(raw_event_size_bytes, DataTransfer::BufferPool::make_bounded(100, 1 * 1024), true), device_(device) {}

V4l2DataTransfer::~V4l2DataTransfer() {}

void V4l2DataTransfer::start_impl(BufferPtr) {
    MV_HAL_LOG_INFO() << "V4l2DataTransfer - start_impl() ";

    buffers = std::make_unique<V4l2DeviceUserPtr>(device_, std::make_unique<DmaBufHeap>("/dev/dma_heap", "linux,cma"),
                                                  8 * 1024 * 1024, 2);

    MV_HAL_LOG_TRACE() << " Nb buffers pre allocated: " << buffers->get_nb_buffers() << std::endl;
    for (unsigned int i = 0; i < buffers->get_nb_buffers(); ++i) {
        buffers->release_buffer(i);
    }
}

void V4l2DataTransfer::run_impl() {
    MV_HAL_LOG_INFO() << "V4l2DataTransfer - run_impl() ";

    while (!should_stop()) {
        // Grab a MIPI frame
        using RawData = DataTransfer::Data *;

        int idx                  = buffers->poll_buffer();
        auto [data, data_length] = buffers->get_buffer_desc(idx);

        MV_HAL_LOG_TRACE() << "Grabed buffer " << idx << "from: " << std::hex << data << " of: " << std::dec
                           << data_length << " Bytes.";

        // Get transfer buffer from the pool and transfer the data
        auto local_buff = get_buffer();
        local_buff->resize(data_length);
        std::memcpy(local_buff->data(), data, data_length);
        transfer_data(local_buff);

        // Reset the buffer data
        memset(data, 0, data_length);

        buffers->release_buffer(idx);
    }
}

void V4l2DataTransfer::stop_impl() {
    MV_HAL_LOG_TRACE() << "V4l2DataTransfer - stop_impl() ";
    buffers.reset();
}

void V4l2DeviceUserPtr::free_buffers() {
    int i = get_nb_buffers();

    while (0 < i) {
        auto idx = poll_buffer();
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

V4l2DeviceUserPtr::V4l2DeviceUserPtr(std::shared_ptr<V4l2Device> device,
                                     std::unique_ptr<Metavision::DmaBufHeap> dma_buf_heap, std::size_t length,
                                     unsigned int nb_buffers) :
    device_(device), dma_buf_heap_(std::move(dma_buf_heap)), length_(length) {
    auto granted_buffers = device->request_buffers(V4L2_MEMORY_USERPTR, nb_buffers);
    MV_HAL_LOG_INFO() << "V4l2 - Requested buffers: " << nb_buffers << " granted buffers: " << granted_buffers.count
                      << std::endl;

    for (unsigned int i = 0; i < granted_buffers.count; ++i) {
        /* Get a buffer using CMA allocator in user space. */
        auto dmabuf_fd = dma_buf_heap_->alloc(length_);

        void *start = mmap(NULL, length_, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
        if (MAP_FAILED == start)
            raise_error("mmap failed");

        dma_buf_heap_->cpu_sync_start(dmabuf_fd);
        memset(start, 0, length_);

        MV_HAL_LOG_TRACE() << "Allocate buffer: " << i << " at: " << std::hex << start << " of " << std::dec << length_
                           << " bytes." << std::endl;

        /* Record the handle to manage the life cycle. */
        buffers_desc_.push_back(BufferDesc{start, dmabuf_fd});
    }
}

V4l2DeviceUserPtr::~V4l2DeviceUserPtr() {
    free_buffers();
}

/** Release the buffer designed by the index to the driver. */
void V4l2DeviceUserPtr::release_buffer(int idx) const {
    auto desc = buffers_desc_.at(idx);

    dma_buf_heap_->cpu_sync_stop(desc.dmabuf_fd);

    V4l2Buffer buf{0};
    buf.type      = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory    = V4L2_MEMORY_USERPTR;
    buf.index     = idx;
    buf.m.userptr = (unsigned long)desc.start;
    buf.length    = length_;
    device_->queue_buffer(buf);
}

/** Poll a MIPI frame buffer through the V4L2 interface.
 * Return the buffer index.
 * */
int V4l2DeviceUserPtr::poll_buffer() const {
    V4l2Buffer buf{0};
    buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_USERPTR;

    while (device_->dequeue_buffer(&buf)) {}

    auto desc = buffers_desc_.at(buf.index);
    dma_buf_heap_->cpu_sync_start(desc.dmabuf_fd);
    return buf.index;
}

/** Return the buffer address and size (in bytes) designed by the index. */
std::pair<void *, std::size_t> V4l2DeviceUserPtr::get_buffer_desc(int idx) const {
    auto desc = buffers_desc_.at(idx);
    return std::make_pair(desc.start, V4l2Device::nb_not_null_data(desc.start, length_));
}
