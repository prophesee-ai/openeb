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
#ifndef DMA_BUF_HEAP_H
#define DMA_BUF_HEAP_H

#include <errno.h>

#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace Metavision {

class DmaBufHeap {
    std::string heap_;
    int heap_fd_;
    std::unordered_set<unsigned int> buffers_fd_;

    void raise_error(const std::string &str) const {
        throw std::runtime_error(str + " (" + std::to_string(errno) + " - " + strerror(errno) + ")");
    }

    std::unordered_set<std::string> get_heap_list(const std::string &heap) {
        std::unordered_set<std::string> heap_list;
        std::unique_ptr<DIR, int (*)(DIR *)> dir(opendir(heap.c_str()), closedir);

        if (dir) {
            struct dirent *dent;
            while ((dent = readdir(dir.get()))) {
                if (!strcmp(dent->d_name, ".") || !strcmp(dent->d_name, ".."))
                    continue;

                heap_list.insert(dent->d_name);
            }
        }

        return heap_list;
    }

    int open_heap(const std::string &heap) {
        int fd = TEMP_FAILURE_RETRY(open(heap.c_str(), O_RDONLY | O_CLOEXEC));
        if (fd < 0)
            raise_error(heap + " Failed to open the heap.");
        return fd;
    }

    static inline int cpu_sync(unsigned int dmabuf_fd, bool start) {
        struct dma_buf_sync sync;
        std::memset(&sync, 0, sizeof(sync));
        sync.flags = (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END) | DMA_BUF_SYNC_RW;

        return TEMP_FAILURE_RETRY(ioctl(dmabuf_fd, DMA_BUF_IOCTL_SYNC, &sync));
    }

public:
    DmaBufHeap(const std::string &heap_path, const std::string &heap_name) : heap_(heap_path + "/" + heap_name) {
        auto heap_list = get_heap_list(heap_path);

        if (heap_list.find(heap_name) == heap_list.end())
            throw std::runtime_error(heap_ + " does not exists.");

        heap_fd_ = open_heap(heap_);
    }
    ~DmaBufHeap() {
        for (auto &buffer_fd : buffers_fd_)
            close(buffer_fd);
        close(heap_fd_);
    }
    static int cpu_sync_start(unsigned int dmabuf_fd) {
        return cpu_sync(dmabuf_fd, true);
    }
    static int cpu_sync_stop(unsigned int dmabuf_fd) {
        return cpu_sync(dmabuf_fd, false);
    }
    unsigned int alloc(size_t len) {
        struct dma_heap_allocation_data heap_data;
        std::memset(&heap_data, 0, sizeof(heap_data));
        heap_data.len      = len;                // length of data to be allocated in bytes
        heap_data.fd_flags = O_RDWR | O_CLOEXEC; // permissions for the memory to be allocated

        auto ret = TEMP_FAILURE_RETRY(ioctl(heap_fd_, DMA_HEAP_IOCTL_ALLOC, &heap_data));
        if (ret < 0) {
            // raise_error(heap_ + " Failed to allocate a buffer of " + std::to_string(len) +
            //            " bytes from: " + std::to_string(heap_fd_));
        } else
            buffers_fd_.insert(heap_data.fd);

        return heap_data.fd;
    }
    void free(unsigned int dmabuf_fd) {
        if (buffers_fd_.find(dmabuf_fd) != buffers_fd_.end())
            close(dmabuf_fd);
    }
};

} // namespace Metavision
#endif // DMA_BUF_HEAP_kkH
