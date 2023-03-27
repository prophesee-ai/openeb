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

#include <cstdint>
#include <string>

#include "metavision/hal/device/device.h"
#include "metavision/hal/facilities/i_digital_crop.h"
#include "metavision/hal/facilities/i_digital_event_mask.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/facilities/i_monitoring.h"
#include "metavision/hal/plugin/plugin.h"
#include "metavision/hal/plugin/plugin_entrypoint.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/file_discovery.h"
#include "metavision/hal/utils/hal_software_info.h"
#include <metavision/hal/utils/camera_discovery.h>

#include "dummy_test_plugin_facilities.h"

using namespace Metavision;

namespace {

struct DummyFileHWIdentification : public I_HW_Identification {
    DummyFileHWIdentification(const std::shared_ptr<I_PluginSoftwareInfo> &plugin_sw_info,
                              const RawFileHeader &header) :
        I_HW_Identification(plugin_sw_info), header_(header) {}

    std::string get_serial() const {
        return std::string();
    }

    long get_system_id() const {
        return 0;
    }

    SensorInfo get_sensor_info() const {
        return SensorInfo({0, 0, "Gen0.0"});
    }

    std::vector<std::string> get_available_data_encoding_formats() const {
        return std::vector<std::string>();
    }

    std::string get_current_data_encoding_format() const {
        return std::string();
    }

    std::string get_integrator() const {
        return header_.get_camera_integrator_name();
    }

    std::string get_connection_type() const {
        return std::string();
    }

    DeviceConfigOptionMap get_device_config_options_impl() const {
        return {};
    }

    RawFileHeader header_;
};

struct DummyFileDiscovery : public FileDiscovery {
    bool discover(DeviceBuilder &device_builder, std::unique_ptr<std::istream> &file, const RawFileHeader &header,
                  const RawFileConfig &file_config) override {
        device_builder.add_facility(
            std::make_unique<DummyFileHWIdentification>(device_builder.get_plugin_software_info(), header));

        return true;
    }
};

struct DummyDigitalEvenMask : public I_DigitalEventMask {
    class DummyPixelMask : public I_PixelMask {
        uint32_t x_{0}, y_{0};
        bool enabled_{false};

        bool set_mask(uint32_t x, uint32_t y, bool enabled) override final {
            x_       = x;
            y_       = y;
            enabled_ = enabled;
            return true;
        }
        std::tuple<uint32_t, uint32_t, bool> get_mask() const override final {
            return std::make_tuple(x_, y_, enabled_);
        }
    };

    DummyDigitalEvenMask() {
        pixel_masks_.push_back(std::make_shared<DummyPixelMask>());
    }
    const std::vector<I_PixelMaskPtr> &get_pixel_masks() const override final {
        return pixel_masks_;
    }

private:
    std::vector<I_PixelMaskPtr> pixel_masks_;
};

struct DummyMonitoring : public I_Monitoring {
    int get_temperature() override final {
        return 12;
    }
    int get_illumination() override final {
        return 34;
    }
    int get_pixel_dead_time() override final {
        return 56;
    }
};

class DummyDigitalCrop : public I_DigitalCrop {
private:
    bool enabled_ = false;
    Region region_;

public:
    bool enable(bool state) override {
        enabled_ = state;
        return true;
    }
    bool is_enabled() override {
        return enabled_;
    }
    bool set_window_region(const Region &region, bool reset_origin) override {
        using std::get;
        if (get<0>(region) > get<2>(region)) {
            throw std::runtime_error("Crop region error");
        }
        if (get<1>(region) > get<3>(region)) {
            throw std::runtime_error("Crop region error");
        }
        region_ = region;
        return true;
    }
    Region get_window_region() override {
        return region_;
    }
};

struct DummyCameraDiscovery : public CameraDiscovery {
    SerialList list() override final {
        return SerialList{"__DummyTest__"};
    }
    SystemList list_available_sources() override final {
        return SystemList{PluginCameraDescription{"__DummyTest__", ConnectionType::PROPRIETARY_LINK, 4321}};
    }
    bool discover(DeviceBuilder &device_builder, const std::string &serial, const DeviceConfig &config) override final {
        device_builder.add_facility(std::make_unique<DummyDigitalCrop>());
        device_builder.add_facility(std::make_unique<DummyDigitalEvenMask>());
        device_builder.add_facility(std::make_unique<DummyMonitoring>());
        device_builder.add_facility(std::make_unique<DummyFacilityV3>());
        return true;
    }

    bool is_for_local_camera() const override final {
        return true;
    }
};

} // namespace

void initialize_plugin(void *plugin_ptr) {
    Metavision::Plugin &plugin = Metavision::plugin_cast(plugin_ptr);
    plugin.set_integrator_name("__DummyTestPlugin__");
    plugin.set_plugin_info(Metavision::get_hal_software_info());
    plugin.set_hal_info(Metavision::get_hal_software_info());

    plugin.add_file_discovery(std::make_unique<DummyFileDiscovery>());
    plugin.add_camera_discovery(std::make_unique<DummyCameraDiscovery>());
}
