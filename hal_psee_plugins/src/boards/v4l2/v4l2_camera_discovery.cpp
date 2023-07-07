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

#include <array>
#include <memory>
#include <string>

//#include "boards/v4l2/dma_buf_heap.h"
#include "boards/v4l2/v4l2_camera_discovery.h"
#include "boards/v4l2/v4l2_data_transfer.h"
#include "boards/v4l2/v4l2_device_mmap.h"
#include "boards/v4l2/v4l2_device_user_ptr.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/plugin/plugin_entrypoint.h"
#include "metavision/hal/utils/camera_discovery.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/psee_hw_layer/utils/psee_format.h"

#include "utils/make_decoder.h"

namespace Metavision {

bool V4l2CameraDiscovery::is_for_local_camera() const {
    return true;
}

CameraDiscovery::SerialList V4l2CameraDiscovery::list() {
    return {"V4l2_device_serial_0"};
}

CameraDiscovery::SystemList V4l2CameraDiscovery::list_available_sources() {
    return {{"V4l2_device_serial_0", ConnectionType::MIPI_LINK, 000420}};
}

bool V4l2CameraDiscovery::discover(DeviceBuilder &device_builder, const std::string &serial,
                                   const DeviceConfig &config) {
    MV_HAL_LOG_TRACE() << "V4l2Discovery - Discovering...";

    std::vector<std::string> device_names = {
        // We might want to iterate over all /dev/videoX files
        "/dev/video0", "/dev/video1", "/dev/video2", "/dev/video3", "/dev/video4",
    };

    std::vector<V4l2Device> devices;
    for (auto device_name : device_names) {
        try {
            /*
            const std::string heap_path = "/dev/dma_heap";
            const std::string heap_name = "linux,cma";

            // We might want to expose the buffer number/size to facilitate tweaking
            constexpr size_t buffer_length = 8 * 1024 * 1024;
            constexpr uint32_t nb_buffers = 32;

            std::unique_ptr<DmaBufHeap> buffer_heap = std::make_unique<DmaBufHeap>(heap_path, heap_name);

            new V4l2DeviceUserPtr(device_name, std::move(buffer_heap), buffer_length, nb_buffers)
            */

            devices.emplace_back(device_name);

        } catch (std::exception &e) {
            MV_HAL_LOG_TRACE() << "Cannot open V4L2 device '" << device_name << "' (err: " << e.what();
        }
    }

    if (devices.empty()) {
        return false;
    }

    class V4l2DeviceControl : public DeviceControl {
    public:
        virtual void start() {
            MV_HAL_LOG_INFO() << "V4l2 Device Control - start()";
        }
        virtual void stop() {
            MV_HAL_LOG_INFO() << "V4l2 Device Control - stop()";
        }
        virtual void reset() {
            MV_HAL_LOG_INFO() << "V4l2 Device Control - reset()";
        }
    };

    class V4l2DataTransfer : public DataTransfer {
    public:
        V4l2DataTransfer(uint32_t raw_event_size_bytes) : DataTransfer(raw_event_size_bytes) {}
        ~V4l2DataTransfer() {}

    private:
        void start_impl(BufferPtr buffer) override final {
            MV_HAL_LOG_INFO() << "V4l2DataTransfer - start_impl() ";
        }
        void run_impl() override final {
            MV_HAL_LOG_INFO() << "V4l2DataTransfer - run_impl() ";
        }
        void stop_impl() override final {
            MV_HAL_LOG_INFO() << "V4l2DataTransfer - start_impl() ";
        }
    };

    auto &main_device = devices[0];

    auto software_info = device_builder.get_plugin_software_info();
    auto hw_id       = device_builder.add_facility(std::make_unique<V4l2HwIdentification>(main_device, software_info));
    auto device_ctrl = std::make_shared<V4l2DeviceControl>();

    try {
        auto encoding_format = StreamFormat(hw_id->get_current_data_encoding_format());
        size_t raw_size_bytes;
        auto decoder = make_decoder(device_builder, encoding_format, raw_size_bytes, false);

        auto v4l2_data_transfer = std::make_unique<V4l2DataTransfer>(raw_size_bytes);

        auto event_stream =
            std::make_unique<Metavision::I_EventsStream>(std::move(v4l2_data_transfer), hw_id, decoder, device_ctrl);
        auto event_stream_facility = device_builder.add_facility(std::move(event_stream));

    } catch (std::exception &e) { MV_HAL_LOG_WARNING() << "System can't stream:" << e.what(); }

    return true;
}

} // namespace Metavision
