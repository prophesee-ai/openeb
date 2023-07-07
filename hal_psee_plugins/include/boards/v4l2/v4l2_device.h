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

#include <algorithm>
#include <string>
#include <linux/videodev2.h>

#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/utils/camera_discovery.h"

namespace Metavision {

void raise_error(const std::string &str);

class V4l2Device {
protected:
    int fd_ = -1;

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

    unsigned int request_buffers(v4l2_memory memory, unsigned int nb_buffers);

    virtual unsigned int get_nb_buffers() const {
        return 0;
    }

public:
    using V4l2Capability = struct v4l2_capability;

    V4l2Device(const std::string &dev_name);
    virtual ~V4l2Device() = default;

    void start();
    void stop();

    virtual void release_buffer(int idx) const {}
    virtual int get_buffer() const {
        return 0;
    }
    virtual std::pair<void *, std::size_t> get_buffer_desc(int idx) const {
        return {nullptr, 0};
    }

    V4l2Capability get_capability() const {
        return cap_;
    }

private:
    V4l2Capability cap_;
};

class V4l2HwIdentification : public I_HW_Identification {
    V4l2Device::V4l2Capability cap_;

public:
    V4l2HwIdentification(const V4l2Device &device, const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info) :
        I_HW_Identification(plugin_sw_info), cap_(device.get_capability()) {}

    // @TODO Retrieve those info through V4L2
    virtual long get_system_id() const {
        return 1234;
    }
    // @TODO Retrieve those info through V4L2
    virtual SensorInfo get_sensor_info() const {
        return {4, 1, "imx636"};
    }
    // @TODO Retrieve those info through V4L2
    virtual std::vector<std::string> get_available_data_encoding_formats() const {
        return {"EVT3", "EVT2"};
    }
    // @TODO Retrieve those info through V4L2
    virtual std::string get_current_data_encoding_format() const {
        return "EVT3;height=720;width=1280";
    }

    virtual std::string get_serial() const {
        std::stringstream ss;
        ss << cap_.card;
        return ss.str();
    }
    virtual std::string get_integrator() const {
        std::stringstream ss;
        ss << cap_.driver;
        return ss.str();
    }
    virtual std::string get_connection_type() const {
        std::stringstream ss;
        ss << cap_.bus_info;
        return ss.str();
    }

protected:
    virtual DeviceConfigOptionMap get_device_config_options_impl() const {
        return {};
    }
};

} // namespace Metavision

#endif // METAVISION_HAL_PSEE_PLUGINS_V4L2_DEVICE_H
