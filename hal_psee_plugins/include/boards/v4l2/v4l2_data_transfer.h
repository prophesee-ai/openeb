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
#include "metavision/sdk/base/utils/object_pool.h"

#include <cstddef>
#include <map>
#include <memory>
#include <memory_resource>
#include <vector>

#include <linux/videodev2.h>

namespace Metavision {
using V4l2Buffer         = struct v4l2_buffer;
using V4l2RequestBuffers = struct v4l2_requestbuffers;

class DmaBufHeap;

class V4l2DataTransfer : public DataTransfer::RawDataProducer {
public:
    // Constructor using MMAP buffers
    V4l2DataTransfer(int fd, uint32_t raw_event_size_bytes);

    // Constructor using DMABUF buffers
    V4l2DataTransfer(int fd, uint32_t raw_event_size_bytes, const std::string &heap_path, const std::string &heap_name);

    ~V4l2DataTransfer();

private:
    static constexpr int device_buffer_number = 32;

    // The memory type currently in use
    const enum v4l2_memory memtype_;

    const int fd_;

    V4l2RequestBuffers request_buffers(uint32_t nb_buffers);

    void start_impl() override final;
    void run_impl(const DataTransfer &data_transfer) override final;
    void stop_impl() override final;

    class V4l2Allocator : public std::pmr::memory_resource {
        size_t buffer_byte_size_{0};

    protected:
        V4l2Allocator(int videodev_fd);

    public:
        size_t max_byte_size() const noexcept {
            return buffer_byte_size_;
        }
        // Get a descriptor usable with V4L2 API from a data pointer
        virtual void fill_v4l2_buffer(void *, V4l2Buffer &) const = 0;
        virtual void begin_cpu_access(void *) const {}
        virtual void end_cpu_access(void *) const {}
    };

    class V4l2MmapAllocator : public V4l2Allocator {
        // MemoryResource interface
        void *do_allocate(std::size_t bytes, std::size_t alignment) override;
        void do_deallocate(void *p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) override;
        bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override;

        // V4l2Allocator interface
        void fill_v4l2_buffer(void *, V4l2Buffer &) const override;

    public:
        V4l2MmapAllocator(int fd);
        ~V4l2MmapAllocator() override;

    private:
        V4l2RequestBuffers request_buffers(uint32_t nb_buffers);
        // The mapping between buffer indices and their memory mapping
        std::vector<void *> mapping_;
        const int fd_;
    };

    class DmabufAllocator : public V4l2Allocator {
        // MemoryResource interface
        void *do_allocate(std::size_t bytes, std::size_t alignment) override;
        void do_deallocate(void *p, std::size_t bytes, std::size_t alignment = alignof(std::max_align_t)) override;
        bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override;

        // V4l2Allocator interface
        void fill_v4l2_buffer(void *, V4l2Buffer &) const override;
        void begin_cpu_access(void *) const override;
        void end_cpu_access(void *) const override;

    public:
        DmabufAllocator(int fd, std::unique_ptr<DmaBufHeap> &&);
        ~DmabufAllocator() override;

    private:
        // The mapping between buffer fds and their memory mapping
        std::map<void *, int> mapping_;
        // Dmabuf heap where the memory is allocated
        std::unique_ptr<DmaBufHeap> dmabuf_heap_;
        // A helper to get the fd from a pointer
        int fd(void *) const;
    };

    std::unique_ptr<V4l2Allocator> v4l2_allocator_;
    using ObjectPool = Metavision::SharedObjectPool<std::pmr::vector<uint8_t>>;
    using BufferPtr  = ObjectPool::ptr_type;
    using Allocator  = ObjectPool::value_type::allocator_type;
    ObjectPool pool_;

    // List of queued buffers to prevent them from going back to the ObjectPool
    BufferPtr queued_buffers_[device_buffer_number];

    // A helper to get the right allocator, and calls to its methods
    V4l2Allocator &v4l2_alloc(BufferPtr &) const;
    void fill_v4l2_buffer(BufferPtr &, V4l2Buffer &) const;
    void begin_cpu_access(BufferPtr &) const;
    void end_cpu_access(BufferPtr &) const;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_V4L2_DATA_TRANSFER_H
