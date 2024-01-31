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

#include <system_error>

#include <unistd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#include "boards/v4l2/v4l2_data_transfer.h"
#include "boards/v4l2/dma_buf_heap.h"

#include "metavision/hal/utils/hal_log.h"

using namespace Metavision;
using Allocator = DataTransfer::Allocator;

V4l2DataTransfer::DmabufAllocator::DmabufAllocator(int fd, std::unique_ptr<DmaBufHeap> &&heap) :
    V4l2Allocator(fd), dmabuf_heap_(std::move(heap)) {}

V4l2DataTransfer::DmabufAllocator::~DmabufAllocator() {}

V4l2DataTransfer::Data *V4l2DataTransfer::DmabufAllocator::allocate(size_t n) {
    void *vaddr;

    if (n > max_size())
        throw std::length_error("Trying to allocate more than the V4L2 buffer length");

    // Alloc a new buffer in the DMA buffer heap
    auto dmabuf_fd = dmabuf_heap_->alloc(n * sizeof(Data));

    // Map it in the program memory
    vaddr = mmap(NULL, n * sizeof(Data), PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
    if (vaddr == MAP_FAILED)
        throw std::system_error(errno, std::generic_category(), "Could not mmap DMABUF buffer");

    // Save the mapping
    mapping_[vaddr] = dmabuf_fd;

    return V4l2DataTransfer::Allocator::pointer(vaddr);
}

void V4l2DataTransfer::DmabufAllocator::deallocate(Data *p, size_t n) {
    // remove buffer mapping in userspace
    munmap((void *)p, n * sizeof(Data));
    // free it in the DmaHeap
    dmabuf_heap_->free(mapping_[p]);
    // Drop the map entry
    mapping_.erase(p);
}

void V4l2DataTransfer::DmabufAllocator::fill_v4l2_buffer(void *vaddr, V4l2Buffer &buf) const {
    buf.m.fd = fd(vaddr);
}

void V4l2DataTransfer::DmabufAllocator::begin_cpu_access(void *vaddr) const {
    dmabuf_heap_->cpu_sync_start(fd(vaddr));
}

void V4l2DataTransfer::DmabufAllocator::end_cpu_access(void *vaddr) const {
    dmabuf_heap_->cpu_sync_stop(fd(vaddr));
}

int V4l2DataTransfer::DmabufAllocator::fd(void *vaddr) const {
    auto it = mapping_.find(vaddr);
    if (it == mapping_.end())
        throw std::system_error(EINVAL, std::generic_category(), "Requested fd of a non-Dmabuf buffer");
    return it->second;
}
