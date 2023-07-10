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

#ifndef METAVISION_HAL_PSEE_PLUGINS_V4L2_DATA_TRANSFER_H
#define METAVISION_HAL_PSEE_PLUGINS_V4L2_DATA_TRANSFER_H

#include "metavision/hal/utils/data_transfer.h"
#include "boards/v4l2/dma_buf_heap.h"

namespace Metavision {

class V4l2Device;

class V4l2DeviceUserPtr; // @TODO Replace with a V4l2 Buffer class interface

class V4l2DataTransfer : public DataTransfer {
public:
    V4l2DataTransfer(std::shared_ptr<V4l2Device> device, uint32_t raw_event_size_bytes);
    ~V4l2DataTransfer();

private:
    std::shared_ptr<V4l2Device> device_;
    std::unique_ptr<V4l2DeviceUserPtr> buffers;

    void start_impl(BufferPtr buffer) override final;
    void run_impl() override final;
    void stop_impl() override final;
};

/** Manage buffer manipulation through the V4L2 interface.
 * In this implementation, buffers are allocated in user space using a dma_buf allocator. This allocator allocates
 * continuous buffers in physical memory which is necessary as buffers are used by DMA without gather/scatter
 * facility.
 */
class DmaBufHeap;

class V4l2DeviceUserPtr {
public:
    V4l2DeviceUserPtr(std::shared_ptr<V4l2Device> device, std::unique_ptr<Metavision::DmaBufHeap> dma_buf_heap,
                      std::size_t length = 8 * 1024 * 1024, unsigned int nb_buffers = 32);

    virtual ~V4l2DeviceUserPtr();

    /** Poll a MIPI frame buffer through the V4L2 interface.
     * Return the buffer index.
     * */
    int get_buffer() const;

    /** Release the buffer designed by the index to the driver. */
    void release_buffer(int idx) const;

    unsigned int get_nb_buffers() const;

    /** Return the buffer address and size (in bytes) designed by the index. */
    std::pair<void *, std::size_t> get_buffer_desc(int idx) const;

    void free_buffers();

private:
    struct BufferDesc {
        void *start;
        unsigned int dmabuf_fd;
    };

    int fd_;
    std::shared_ptr<V4l2Device> device_;
    std::unique_ptr<DmaBufHeap> dma_buf_heap_;
    std::size_t length_;
    std::vector<BufferDesc> buffers_desc_;

    void allocate_buffers(unsigned int nb_buffers);
};
} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_V4L2_DATA_TRANSFER_H
