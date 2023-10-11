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

// Example of using Metavision SDK Base and Metavision HAL APIs to run a system, platform and software diagnosis.

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <array>
#ifdef __linux__
#include <regex>
#endif
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/base/utils/software_info.h>
#include <metavision/hal/facilities/i_hal_software_info.h>
#include <metavision/hal/facilities/i_plugin_software_info.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_geometry.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/utils/hal_exception.h>

namespace po = boost::program_options;

static constexpr int label_size = 50;

struct ExecState {
    std::string cmd_result = "";
    bool cmd_success       = true;

    operator bool() {
        return cmd_success;
    }
};

#ifdef __linux__
// https://stackoverflow.com/questions/478898/how-to-execute-a-command-and-get-output-of-command-within-c-using-posix
ExecState exec(const char *cmd) {
    static constexpr size_t read_result_buffer_size = 128;
    std::array<char, read_result_buffer_size> buffer;
    ExecState result;
    errno      = 0;
    FILE *pipe = popen(cmd, "r");
    if (!pipe) {
        result.cmd_success = false;
        return result;
    }

    while (!feof(pipe)) {
        if (fgets(buffer.data(), read_result_buffer_size, pipe) != nullptr) {
            result.cmd_result += buffer.data();
        }
    }
    auto find_ind = result.cmd_result.find_last_of("\n");
    if (find_ind != std::string::npos) {
        result.cmd_result.erase(find_ind);
    };
    int ret = pclose(pipe);
    if (ret != 0) {
        result.cmd_success = false;
    }

    return result;
}
#endif

std::string get_geometry(Metavision::Device *device) {
    Metavision::I_Geometry *geometry = device->get_facility<Metavision::I_Geometry>();
    if (geometry) {
        int width  = geometry->get_width();
        int height = geometry->get_height();

        if (width == 480 && height == 360) {
            return "HVGA";
        } else if (width == 1280 && height == 720) {
            return "HD";
        } else if (width == 640 && height == 480) {
            return "VGA";
        } else if (width == 304 && height == 240) {
            return "QVGA";
        } else {
            return std::to_string(width) + "x" + std::to_string(height);
        }

    } else {
        return "Unknown geometry";
    }
}

void print_title(const std::string &title) {
    MV_LOG_INFO();
    MV_LOG_INFO() << "------------------------------------------";
    MV_LOG_INFO() << title;
    MV_LOG_INFO() << "------------------------------------------";
    MV_LOG_INFO();
}

void print_section(const std::string &section) {
    MV_LOG_INFO();
    MV_LOG_INFO() << "#### " << section << " ####";
    MV_LOG_INFO();
}

struct PackageInfo {
    PackageInfo(const std::string &n, const std::string &v, const std::string &a, const std::string &d) :
        name(n), version(v), architecture(a), description(d) {}
    std::string name;
    std::string version;
    std::string architecture;
    std::string description;
};

#ifdef __linux__
bool print_installed_packages(const std::string &reg) {
    std::string command = "dpkg -l | grep " + reg;
    ExecState ret       = exec(command.c_str());
    if (ret) {
        std::istringstream lines(ret.cmd_result);
        std::string line;
        std::regex base_regex("ii\\s+(\\S+)\\s+(\\S+)\\s+(\\S+)\\s+(.+)");
        std::smatch base_match;
        std::vector<PackageInfo> packages;

        while (std::getline(lines, line)) {
            if (std::regex_match(line, base_match, base_regex)) {
                packages.emplace_back(base_match[1].str(), base_match[2].str(), base_match[3].str(),
                                      base_match[4].str());
            }
        }

        if (packages.size() == 0) {
            MV_LOG_INFO() << "None";
        } else {
            MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(35) << "Name" << std::setw(10)
                          << "Version" << std::setw(15) << "Architecture" << std::setw(20) << "Description";
            for (auto &p : packages) {
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(35) << p.name << std::setw(10)
                              << p.version << std::setw(15) << p.architecture << std::setw(20) << p.description;
            }
        }
    } else {
        return false;
    }
    return true;
}
#endif

void do_short_diagnosis() {
    auto serial_list = Metavision::DeviceDiscovery::list();
    if (serial_list.empty()) {
        MV_LOG_ERROR() << "No Device Found";
    }

    for (const auto &serial : serial_list) {
        MV_LOG_INFO() << "Trying to open serial" << serial;
        std::unique_ptr<Metavision::Device> device;

        try {
            // open device from a serial
            device = Metavision::DeviceDiscovery::open(serial);
        } catch (const Metavision::HalException &e) { MV_LOG_ERROR() << e.what(); }

        if (device) {
            auto i_hal_software_info = device->get_facility<Metavision::I_HALSoftwareInfo>();
            if (i_hal_software_info) {
                auto &hal_software_info = i_hal_software_info->get_software_info();
                MV_LOG_INFO() << "## HAL Software";
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                              << "Version:" << hal_software_info.get_version();
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                              << "VCS branch:" << hal_software_info.get_vcs_branch();
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                              << "VCS commit:" << hal_software_info.get_vcs_commit();
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                              << "VCS commit's date:" << hal_software_info.get_vcs_date() << "\n";
            }

            auto i_plugin_software_info = device->get_facility<Metavision::I_PluginSoftwareInfo>();
            if (i_plugin_software_info) {
                auto &plugin_software_info = i_plugin_software_info->get_software_info();
                MV_LOG_INFO() << "## Plugin Software";
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                              << "Name:" << i_plugin_software_info->get_plugin_name();
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                              << "Version:" << plugin_software_info.get_version();
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                              << "VCS branch:" << plugin_software_info.get_vcs_branch();
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                              << "VCS commit:" << plugin_software_info.get_vcs_commit();
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                              << "VCS commit's date:" << plugin_software_info.get_vcs_date() << "\n";
            }

            // Retrieves the facility that provides information about the hardware
            Metavision::I_HW_Identification *hw_identification =
                device->get_facility<Metavision::I_HW_Identification>();
            if (hw_identification != nullptr) {
                MV_LOG_INFO() << "## Hardware";
                // Retrieves a map of key/value with the information
                for (auto system_info : hw_identification->get_system_info()) {
                    auto key            = system_info.first;
                    auto value          = system_info.second;
                    const std::string s = key + ":";
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size) << s << value;
                }
            }
        }
    }
}

void do_software_diagnosis() {
    print_title("METAVISION SOFTWARE INFORMATION");
    print_section("INSTALLED SOFTWARE");

    auto &metavision_sdk_software_info = Metavision::get_metavision_software_info();
    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                  << "Version: " << metavision_sdk_software_info.get_version() << std::right;
    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                  << "Id: " << metavision_sdk_software_info.get_vcs_commit() << std::right;
#ifdef __linux__
    print_section("INSTALLED PACKAGES - METAVISION");

    if (!print_installed_packages("metavision-")) {
        MV_LOG_WARNING() << "Failed to retrieve installed Metavision packages list";
    }
#endif
}

void do_systems_diagnosis() {
    std::string command;
    ExecState ret;

    print_title("METAVISION SYSTEMS INFORMATION");
    print_section("SYSTEMS AVAILABLE");

    auto serial_list = Metavision::DeviceDiscovery::list();

#ifdef __linux
    bool do_usb_port_analysis = false;
#endif
    for (auto s : serial_list) {
        // Open the camera
        std::unique_ptr<Metavision::Device> device(Metavision::DeviceDiscovery::open(s));

        if (device) {
            Metavision::I_HW_Identification *hw_identification =
                device->get_facility<Metavision::I_HW_Identification>();
            if (hw_identification != nullptr) {
                const auto &sensor_info = hw_identification->get_sensor_info();
                MV_LOG_INFO() << "##" << hw_identification->get_integrator() << sensor_info.name_
                              << get_geometry(device.get()) << "##";

                MV_LOG_INFO();

                MV_LOG_INFO() << "# System information";
                // Retrieves a map of key/value with the information
                for (auto system_info : hw_identification->get_system_info()) {
                    auto key   = system_info.first;
                    auto value = system_info.second;
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size) << key << value
                                  << std::right;
                }
                MV_LOG_INFO() << "";

                // Get device config options
                auto options = hw_identification->get_device_config_options();

                MV_LOG_INFO() << "# Available device config options";
                for (auto option : options) {
                    MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size) << option.first
                                  << option.second;
                }
                MV_LOG_INFO() << "";

                Metavision::I_LL_Biases *i_ll_biases = device->get_facility<Metavision::I_LL_Biases>();
                if (i_ll_biases != nullptr) {
                    MV_LOG_INFO();
                    MV_LOG_INFO() << "# Default Biases";
                    auto all_biases = i_ll_biases->get_all_biases();
                    for (auto it = all_biases.begin(), it_end = all_biases.end(); it != it_end; ++it) {
                        std::string bias_name = it->first;
                        int bias_value        = it->second;
                        MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size) << bias_name
                                      << bias_value << std::right;
                    }
                }
            }

#ifdef __linux
            do_usb_port_analysis = true;
#endif
            device.reset(nullptr);
            MV_LOG_INFO() << "";
        }
    }

#ifdef __linux__
    if (do_usb_port_analysis) {
        print_section("SYSTEM'S USB PORTS");

        std::vector<std::string> vendor_and_product_ids = {"04b4:00f4", "04b4:00f5", "03fd:5832"};

        bool usb2_found           = false;
        int nr_usb_port_found     = 0;
        std::string usb_ports_str = "";
        for (auto &vendor_and_product_id : vendor_and_product_ids) {
            command = "lsusb -d " + vendor_and_product_id + " -v 2> /dev/null | grep bcdUSB | awk '{print $2}'";
            ret     = exec(command.c_str());
            if (ret) {
                if (ret.cmd_result != "") {
                    ++nr_usb_port_found;
                    std::istringstream ss(ret.cmd_result);
                    std::string usb_port;
                    while (std::getline(ss, usb_port)) {
                        try {
                            float usb_port_type = std::stof(usb_port);
                            usb2_found |= (usb_port_type < 3.);
                        } catch (...) {}

                        usb_ports_str += usb_port + " ";
                    }
                }
            } else {
                nr_usb_port_found = -1;
                break;
            }
        }
        if (nr_usb_port_found == 0) {
            MV_LOG_INFO() << "No systems USB connected have been found on your platform.";
        } else if (nr_usb_port_found < 0) {
            MV_LOG_WARNING() << Metavision::Log::no_space << std::left << std::setw(label_size) << "USB Port type: "
                             << "Failed to retrieve USB port type" << std::right;
        } else {
            MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                          << "USB Port type used: " << usb_ports_str;
            if (usb2_found) {
                MV_LOG_WARNING() << "You must use a USB 3+ port for optimal performance with a system.";
            }
        }
    } else {
        MV_LOG_INFO() << "No systems USB connected have been found on your platform.";
    }
#endif
}

void do_platform_diagnosis() {
#ifdef __linux__
    std::string command;
    ExecState ret;

    print_title("PLATFORM INFORMATION");
    print_section("DISTRIBUTION");

    command = "uname -a";
    ret     = exec(command.c_str());
    if (ret) {
        MV_LOG_INFO() << ret.cmd_result;
    } else {
        MV_LOG_INFO() << "!!! Failed to retrieve distribution information";
    }

    print_section("OPENGL");
    uid_t user_id                     = getuid();
    std::string command_prefix        = "";
    bool privileged_command_available = true;
    if (user_id != 0) {
        command_prefix = "sudo";
        command        = command_prefix + " echo > /dev/null 2>&1";
        ret            = exec(command.c_str());
        if (!ret) {
            privileged_command_available = false;
            MV_LOG_WARNING() << "Failed to get root access";
        }
    }

    if (ret) {
        command = "glxinfo |grep OpenGL";
        ret     = exec(command.c_str());
        if (ret) {
            std::stringstream ss(ret.cmd_result);
            std::string to;
            while (std::getline(ss, to, '\n')) {
                MV_LOG_INFO() << to;
            }
        } else {
            MV_LOG_WARNING() << "Failed to retrieve OpenGL information";
        }
    } else {
        MV_LOG_WARNING() << "Failed to retrieve OpenGL information";
    }

    print_section("VIRTUAL MACHINE");

    static const std::vector<std::string> virtualization_technology_map = {
        "VMware",     // VMware
        "VirtualBox", // VirtualBox
        "KVM",        // Qemu with KVM
        "Bochs",      // Qemu
        "Microsoft",  // Microsoft virtual PC
    };

    bool on_vm = false;
    if (privileged_command_available) {
        for (auto virtualization_technology : virtualization_technology_map) {
            command = command_prefix + " dmidecode | grep " + virtualization_technology;
            ret     = exec(command.c_str());
            if (ret && ret.cmd_result != "") {
                MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                              << "Virtual Machine:" << Metavision::Log::no_space << "YES [" << ret.cmd_result << "]"
                              << std::right;
                on_vm = true;
                break;
            }
        }
    }

    if (!on_vm) {
        command = "cat /proc/cpuinfo | grep hypervisor";
        ret     = exec(command.c_str());
        if (ret && ret.cmd_result != "") {
            MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                          << "Virtual Machine:" << Metavision::Log::no_space << "YES [Name not found]" << std::right;
        } else {
            MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size)
                          << "Virtual Machine:" << Metavision::Log::no_space << "NO" << std::right;
        }
    }

    if (boost::filesystem::exists(boost::filesystem::path("/.dockerenv"))) {
        MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size) << "Docker:"
                      << "YES" << std::right;
    } else {
        MV_LOG_INFO() << Metavision::Log::no_space << std::left << std::setw(label_size) << "Docker:"
                      << "NO" << std::right;
    }

#endif
}

int main(int argc, char *argv[]) {
    bool display_short_info    = false;
    bool display_systems_info  = false;
    bool display_software_info = false;
    bool display_platform_info = false;
    std::string output_file;

    const std::string program_desc("Metavision diagnosis tool.\n"
                                   "Executes a full diagnosis on Metavision software and systems and check the\n"
                                   "compatibility with your platform.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("short",     po::bool_switch(&display_short_info)->default_value(false), "Display short diagnosis.")
        ("system",    po::bool_switch(&display_systems_info)->default_value(false), "Display system diagnosis.")
        ("software",  po::bool_switch(&display_software_info)->default_value(false), "Display installed software diagnosis.")
        ("platform",  po::bool_switch(&display_platform_info)->default_value(false), "Display platform diagnosis.")
        ("log,l",     po::value<std::string>(&output_file), "Log diagnosis into a file.")
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

    std::ofstream ofs;
    if (!output_file.empty()) {
        ofs.open(output_file);
        Metavision::setLogStream(ofs);
    }

    if (display_short_info && (display_systems_info | display_software_info | display_platform_info)) {
        MV_LOG_ERROR() << "Argument --short can not be used along with --system, --software or --platform";
        return 1;
    }

    if (display_short_info) {
        do_short_diagnosis();
    }

    bool display_all = !(display_short_info | display_systems_info | display_software_info | display_platform_info);

    display_systems_info |= display_all;
    display_software_info |= display_all;
    display_platform_info |= display_all;

    if (display_platform_info) {
        do_platform_diagnosis();
    }

    if (display_software_info) {
        do_software_diagnosis();
    }

    if (display_systems_info) {
        do_systems_diagnosis();
    }

    MV_LOG_INFO();

    return 0;
}
