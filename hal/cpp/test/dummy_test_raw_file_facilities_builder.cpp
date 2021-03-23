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

#include "metavision/hal/utils/file_discovery.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/facilities/i_hw_identification.h"
#include "metavision/hal/plugin/plugin.h"
#include "metavision/hal/plugin/plugin_entrypoint.h"
#include "metavision/hal/device/device.h"
#include "metavision/hal/utils/hal_software_info.h"

namespace {

struct FileHWIdentification : public Metavision::I_HW_Identification {
    FileHWIdentification(const std::shared_ptr<Metavision::I_PluginSoftwareInfo> &plugin_sw_info,
                         const Metavision::RawFileHeader &header) :
        Metavision::I_HW_Identification(plugin_sw_info), header_(header) {}

    std::string get_serial() const {
        return std::string();
    }

    long get_system_id() const {
        return 0;
    }

    SensorInfo get_sensor_info() const {
        return SensorInfo();
    }

    long get_system_version() const {
        return 0;
    }

    std::vector<std::string> get_available_raw_format() const {
        return std::vector<std::string>();
    }

    std::string get_integrator() const {
        return header_.get_integrator_name();
    }

    std::string get_connection_type() const {
        return std::string();
    }

    Metavision::RawFileHeader header_;
};

struct FileDiscovery : public Metavision::FileDiscovery {
    bool discover(Metavision::DeviceBuilder &device_builder, std::unique_ptr<std::istream> &file,
                  const Metavision::RawFileHeader &header, const Metavision::RawFileConfig &file_config) override {
        device_builder.add_facility(
            std::make_unique<FileHWIdentification>(device_builder.get_plugin_software_info(), header));

        return true;
    }
};
} // namespace

void initialize_plugin(void *plugin_ptr) {
    Metavision::Plugin &plugin = Metavision::plugin_cast(plugin_ptr);
    plugin.set_integrator_name("__DummyTest__");
    plugin.set_plugin_info(Metavision::get_hal_software_info());
    plugin.set_hal_info(Metavision::get_hal_software_info());

    plugin.add_file_discovery(std::make_unique<FileDiscovery>());
}
