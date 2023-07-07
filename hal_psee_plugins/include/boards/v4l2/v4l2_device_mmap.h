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

#ifndef METAVISION_HAL_PSEE_PLUGINS_V4L2_DEVICE_NMAP_H
#define METAVISION_HAL_PSEE_PLUGINS_V4L2_DEVICE_NMAP_H

#include "boards/v4l2/v4l2_device.h"

namespace Metavision {

/** Manage buffer manipulation through the V4L2 interface.
 * In this implementation, buffers are allocated in the driver after a request during the .
 */
class V4l2DeviceMmap : public V4l2Device {
public:
    V4l2DeviceMmap(const char *dev_name, unsigned int nb_buffers = 32);
    ~V4l2DeviceMmap();

    /** Release the buffer designed by the index to the driver. */
    void release_buffer(int idx) const final;

    /** Poll a MIPI frame buffer through the V4L2 interface.
     * Return the buffer index.
     * */
    int get_buffer() const final;

    /** Return the buffer address and size (in bytes) designed by the index. */
    std::pair<void *, std::size_t> get_buffer_desc(int idx) const final;

private:
    struct BufferDesc {
        void *start;
        std::size_t length; /* In bytes. */
    };

    std::vector<BufferDesc> buffers_desc_;

    void query_buffers(unsigned int nb_buffers);
    void free_buffers();
    unsigned int get_nb_buffers() const final;
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_V4L2_DEVICE_NMAP_H
