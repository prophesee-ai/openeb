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

#ifndef METAVISION_HAL_PSEE_PLUGINS_V4L2_DEVICE_H
#define METAVISION_HAL_PSEE_PLUGINS_V4L2_DEVICE_H

#include <string>
#include <vector>

#include <linux/videodev2.h>

#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/hal/utils/camera_discovery.h"
#include "metavision/hal/utils/device_control.h"

namespace Metavision {

void raise_error(const std::string &str);

using V4l2Capability     = struct v4l2_capability;
using V4l2Buffer         = struct v4l2_buffer;
using V4l2RequestBuffers = struct v4l2_requestbuffers;

class V4L2DeviceControl : public DeviceControl {
    V4l2Capability cap_;
    int fd_ = -1;

public:
    template<class Data>
    static typename std::vector<uint64_t>::const_iterator iterator_cast(Data *ptr) {
        return typename std::vector<uint64_t>::const_iterator(reinterpret_cast<const uint64_t *>(ptr));
    }

    /* Count the number of bytes received in the buffer. The complexity is log(n) */
    template<typename Data>
    static std::size_t nb_not_null_data(const Data *const buf_beg_addr, std::size_t length_in_bytes) {
        auto is_not_null = [](const auto &d) { return d != 0; };
        auto beg         = iterator_cast(buf_beg_addr);
        auto end         = beg + length_in_bytes / sizeof(*beg);

        auto it_pp = std::partition_point(beg, end, is_not_null);
        return std::distance(beg, it_pp) * sizeof(*beg);
    }

    V4L2DeviceControl(const std::string &dev_name);
    virtual ~V4L2DeviceControl() = default;

    V4l2Capability get_capability() const;

    V4l2RequestBuffers request_buffers(v4l2_memory memory, uint32_t nb_buffers);
    V4l2Buffer query_buffer(v4l2_memory memory_type, uint32_t buf_index);
    int queue_buffer(V4l2Buffer &buffer);
    int dequeue_buffer(V4l2Buffer *buffer);
    int get_fd() const {
        return fd_;
    }

    // DeviceControl
public:
    virtual void start() override;
    virtual void stop() override;
    virtual void reset() override;
};

class V4l2Synchronization : public I_CameraSynchronization {
public:
    virtual bool set_mode_standalone() override {
        return true;
    }
    virtual bool set_mode_master() override {
        return false;
    }
    virtual bool set_mode_slave() override {
        return false;
    }
    virtual SyncMode get_mode() const override {
        return SyncMode::STANDALONE;
    }
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_V4L2_DEVICE_H
