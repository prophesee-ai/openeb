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

using namespace Metavision;

V4l2DataTransfer::V4l2MmapAllocator::V4l2MmapAllocator(int fd) :
    V4l2Allocator(fd), mapping_(device_buffer_number, nullptr), fd_(dup(fd)) {}

V4l2DataTransfer::V4l2MmapAllocator::~V4l2MmapAllocator() {
    close(fd_);
}

void *V4l2DataTransfer::V4l2MmapAllocator::do_allocate(std::size_t bytes, std::size_t alignment) {
    void *vaddr;
    int buffer_index;

    if (bytes > max_byte_size())
        throw std::length_error("Trying to expand allocation beyond V4L2 buffer length");

    // Look for a free buffer
    for (buffer_index = 0; buffer_index < device_buffer_number; buffer_index++)
        if (mapping_[buffer_index] == nullptr)
            break;
    if (buffer_index >= device_buffer_number)
        throw std::system_error(ENOMEM, std::generic_category(), "No more available V4L2 buffer");

    // Query buffer information
    V4l2Buffer buffer{};
    buffer.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.index  = buffer_index;

    if (ioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0)
        throw std::system_error(errno, std::generic_category(), "Could not query V4L2 buffer");

    // Map it in the program memory
    vaddr = mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buffer.m.offset);
    if (vaddr == MAP_FAILED)
        throw std::system_error(errno, std::generic_category(), "Could not mmap V4L2 buffer");

    // Save the mapping, implicitly making the buffer used.
    mapping_[buffer_index] = vaddr;

    return vaddr;
}

void V4l2DataTransfer::V4l2MmapAllocator::do_deallocate(void *p, std::size_t bytes, std::size_t alignment) {
    // Mark the buffer as unused
    for (int i = 0; i < device_buffer_number; i++)
        if (mapping_[i] == p)
            mapping_[i] = nullptr;
    // and remove its mapping in userspace
    munmap(p, max_byte_size());
}

bool V4l2DataTransfer::V4l2MmapAllocator::do_is_equal(const std::pmr::memory_resource &other) const noexcept {
    return dynamic_cast<const V4l2MmapAllocator *>(&other) != nullptr;
}

void V4l2DataTransfer::V4l2MmapAllocator::fill_v4l2_buffer(void *vaddr, V4l2Buffer &buf) const {
    // There are at most 32 buffers, a std::map looks overkill for index search
    for (int i = 0; i < device_buffer_number; i++)
        if (mapping_[i] == vaddr) {
            buf.index = i;
            return;
        }
    throw std::system_error(EINVAL, std::generic_category(), "Requested index of a non-V4L2 buffer");
}
