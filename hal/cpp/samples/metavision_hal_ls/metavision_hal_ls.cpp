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

#include <exception>
#include <iostream>
#include <iomanip>
#include <boost/program_options.hpp>

#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/facilities/i_hal_software_info.h>
#include <metavision/hal/facilities/i_plugin_software_info.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/utils/hal_exception.h>
#include <metavision/sdk/base/utils/log.h>

namespace po = boost::program_options;
int main(int argc, char *argv[]) {
    const std::string program_desc(
        "This code sample demonstrates how to use Metavision HAL to enumerate information about "
        "devices connected to the machine in the following order: <integrator>:<plugin>:<serial>.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("verbose,v", "Print more information.")
        ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    bool verbose = false;
    if (vm.count("verbose")) {
        verbose = true;
    }

    /// [Retrieves serial numbers]
    auto v = Metavision::DeviceDiscovery::list();

    if (v.empty()) {
        MV_LOG_ERROR() << "No device found";
    }
    /// [Retrieves serial numbers]

    for (auto s : v) {
        MV_LOG_INFO() << "Device detected:" << s;
        if (verbose) {
            std::unique_ptr<Metavision::Device> device;

            try {
                /// [open serial]
                // open device from a serial
                device = Metavision::DeviceDiscovery::open(s);
                /// [open serial]
            } catch (const Metavision::HalException &e) { MV_LOG_ERROR() << e.what(); }

            if (device) {
                auto i_hal_software_info = device->get_facility<Metavision::I_HALSoftwareInfo>();
                if (i_hal_software_info) {
                    auto &hal_software_info = i_hal_software_info->get_software_info();
                    MV_LOG_INFO() << "## HAL Software";
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(30)
                                  << "Version:" << hal_software_info.get_version();
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(30)
                                  << "VCS branch:" << hal_software_info.get_vcs_branch();
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(30)
                                  << "VCS commit:" << hal_software_info.get_vcs_commit();
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(30)
                                  << "VCS commit's date:" << hal_software_info.get_vcs_date() << "\n";
                }

                auto i_plugin_software_info = device->get_facility<Metavision::I_PluginSoftwareInfo>();
                if (i_plugin_software_info) {
                    auto &plugin_software_info = i_plugin_software_info->get_software_info();
                    MV_LOG_INFO() << "## Plugin Software";
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(30)
                                  << "Name:" << i_plugin_software_info->get_plugin_name();
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(30)
                                  << "Version:" << plugin_software_info.get_version();
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(30)
                                  << "VCS branch:" << plugin_software_info.get_vcs_branch();
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(30)
                                  << "VCS commit:" << plugin_software_info.get_vcs_commit();
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(30)
                                  << "VCS commit's date:" << plugin_software_info.get_vcs_date() << "\n";
                }

                /// [get facility]
                // Retrieves the facility that provides information about the hardware
                Metavision::I_HW_Identification *hw_identification =
                    device->get_facility<Metavision::I_HW_Identification>();
                if (hw_identification) {
                    MV_LOG_INFO() << "## Hardware";
                    // Retrieves a map of key/value with the information
                    for (auto system_info : hw_identification->get_system_info()) {
                        auto key            = system_info.first;
                        auto value          = system_info.second;
                        const std::string s = key + ":";
                        MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(30) << s << value;
                    }
                }
                /// [get facility]
            }
        }
    }
    return 0;
}
