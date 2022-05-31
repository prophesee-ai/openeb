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

#include "metavision/hal/utils/detail/hal_log_impl.h"
#include <memory>
#include <assert.h>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <dirent.h>
#ifdef _WIN32
#include <windows.h>
#include <strsafe.h>
#else
#include <dlfcn.h>
#endif

#include "metavision/hal/device/device.h"
#include "metavision/hal/utils/device_builder.h"
#include "metavision/hal/utils/raw_file_header.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/facilities/future/i_events_stream.h"
#include "metavision/hal/facilities/i_hal_software_info.h"
#include "metavision/hal/facilities/i_plugin_software_info.h"
#include "metavision/hal/device/device_discovery.h"
#include "metavision/hal/utils/camera_discovery.h"
#include "metavision/hal/utils/file_discovery.h"
#include "metavision/hal/utils/device_config.h"
#include "metavision/hal/facilities/i_events_stream.h"
#include "metavision/hal/utils/hal_error_code.h"
#include "metavision/hal/utils/hal_exception.h"
#include "metavision/hal/utils/hal_log.h"
#include "metavision/hal/utils/resources_folder.h"
#include "metavision/hal/plugin/plugin.h"
#include "metavision/hal/plugin/detail/plugin_loader.h"

namespace {

// Type of camera to be returned
enum CameraType {
    REMOTE = 1,
    LOCAL  = 2,
    ANY    = 3,
};
std::string CameraTypeLabels[] = {"remote", "local", "any"};

Metavision::PluginLoader plugin_loader;

Metavision::PluginLoader::PluginList get_plugins() {
    static bool loaded            = false;
    static char *last_plugin_path = nullptr;

    MV_HAL_LOG_TRACE() << "Loading plugins";

    char *plugin_path = getenv("MV_HAL_PLUGIN_PATH");
    if (loaded && (!plugin_path || strcmp(plugin_path, last_plugin_path) == 0)) {
        MV_HAL_LOG_TRACE()
            << "  MV_HAL_PLUGIN_PATH did not change and plugins are already loaded, no need to reload plugins";
        return plugin_loader.get_plugin_list();
    }
    last_plugin_path = plugin_path;

    plugin_loader.clear_folders();
    MV_HAL_LOG_TRACE() << "  Setting up search paths";
    if (plugin_path) {
        std::string plugin_folders(plugin_path);
#ifdef _WIN32
        std::string delimiter = ";";
#else
        std::string delimiter = ":";
#endif
        size_t pos = 0;
        std::string folder;
        std::vector<std::string> folders;
        while ((pos = plugin_folders.find(delimiter)) != std::string::npos) {
            folder = plugin_folders.substr(0, pos);
            MV_HAL_LOG_TRACE() << "    Adding plugin search path:" << folder;
            folders.push_back(folder);
            plugin_folders.erase(0, pos + delimiter.length());
        }
        if (!plugin_folders.empty()) {
            MV_HAL_LOG_TRACE() << "    Adding plugin search path:" << plugin_folders;
            folders.push_back(plugin_folders);
        }
        plugin_loader.insert_folders(folders);
        plugin_loader.load_plugins();
        plugin_loader.clear_folders();
    }

    // Insert installation path to plugin folders
    // Remark : we do it here (after adding folders from MV_HAL_PLUGIN_PATH)
    // because we want to look first in env var MV_HAL_PLUGIN_PATH (if set by the user)
    // and then in the installation path
    std::string plugin_install_path = Metavision::ResourcesFolder::get_plugin_install_path();
    if (!plugin_install_path.empty()) {
        MV_HAL_LOG_TRACE() << "    Adding plugin search path:" << plugin_install_path;
        plugin_loader.insert_folder(plugin_install_path);
        plugin_loader.load_plugins();
        plugin_loader.clear_folders();
    }

    MV_HAL_LOG_TRACE() << "  Loading plugins...";
    bool has_camera_discovery = false;
    bool has_file_discovery   = false;
    auto plugin_list          = plugin_loader.get_plugin_list();
    for (auto &plugin : plugin_list) {
        if (plugin.get_camera_discovery_list().size() != 0) {
            has_camera_discovery = true;
        }
        if (plugin.get_file_discovery_list().size() != 0) {
            has_file_discovery = true;
        }
        MV_HAL_LOG_TRACE() << Metavision::Log::no_space << "    [" << plugin.get_plugin_name() << "] ("
                           << plugin.get_integrator_name() << ") " << plugin.get_camera_discovery_list().size()
                           << " camera discoveries " << plugin.get_file_discovery_list().size() << " file discoveries";
    }

    if (!has_camera_discovery || !has_file_discovery) {
        if (plugin_list.empty()) {
            MV_HAL_LOG_WARNING() << "    no plugin found";
        } else if (!has_camera_discovery && !has_file_discovery) {
            MV_HAL_LOG_WARNING() << "    no plugin provides either camera or file discovery functionnality";
        } else if (!has_camera_discovery) {
            MV_HAL_LOG_TRACE() << "    no plugin provides camera discovery functionnality";
        } else if (!has_file_discovery) {
            MV_HAL_LOG_TRACE() << "    no plugin provides file discovery functionnality";
        }
    } else {
        MV_HAL_LOG_TRACE() << "  Found" << plugin_list.size() << "plugins";
    }
    loaded = true;

    return plugin_loader.get_plugin_list();
}

std::string get_full_serial(const std::string &integrator, const std::string &plugin, const std::string &serial) {
    return integrator + ":" + plugin + ":" + serial;
}

void common_log_plugin_error(const Metavision::Plugin &plugin, const std::string &discovery_name) {
    MV_HAL_LOG_ERROR() << "Error while opening from plugin:" << plugin.get_plugin_name();
    MV_HAL_LOG_ERROR() << "Integrator:" << plugin.get_integrator_name();
    MV_HAL_LOG_ERROR() << "Device discovery:" << discovery_name;
}

void log_plugin_error(const Metavision::Plugin &plugin, const std::string &discovery_name,
                      const Metavision::HalException &e) {
    common_log_plugin_error(plugin, discovery_name);
    MV_HAL_LOG_ERROR() << "Failed with exception:";
    MV_HAL_LOG_ERROR() << e.what();
}

void log_plugin_error(const Metavision::Plugin &plugin, const std::string &discovery_name, const std::exception &e) {
    common_log_plugin_error(plugin, discovery_name);
    MV_HAL_LOG_ERROR() << "Failed with non Metavision HAL default exception:";
    MV_HAL_LOG_ERROR() << e.what();
}

void log_plugin_error(const Metavision::Plugin &plugin, const std::string &discovery_name) {
    common_log_plugin_error(plugin, discovery_name);
    MV_HAL_LOG_ERROR() << "Failed with non Metavision HAL default exception:";
}
} // anonymous namespace

namespace Metavision {

std::string CameraDescription::get_full_serial() const {
    return ::get_full_serial(integrator_name_, plugin_name_, serial_);
}

bool operator==(const PluginCameraDescription &lhs, const PluginCameraDescription &rhs) {
    return lhs.serial_ == rhs.serial_ && lhs.connection_ == rhs.connection_ && lhs.system_id_ == rhs.system_id_;
}
bool operator!=(const PluginCameraDescription &lhs, const PluginCameraDescription &rhs) {
    return !(lhs == rhs);
}

bool operator==(const CameraDescription &lhs, const CameraDescription &rhs) {
    return PluginCameraDescription(lhs) == PluginCameraDescription(rhs) &&
           lhs.integrator_name_ == rhs.integrator_name_ && lhs.plugin_name_ == rhs.plugin_name_;
}
bool operator!=(const CameraDescription &lhs, const CameraDescription &rhs) {
    return !(lhs == rhs);
}

DeviceDiscovery::SerialList list_serial_camera(CameraType flag = ANY) {
    DeviceDiscovery::SerialList ret;

    MV_HAL_LOG_TRACE() << "Listing cameras of" << CameraTypeLabels[flag - 1] << "type";

    for (auto &plugin : get_plugins()) {
        MV_HAL_LOG_TRACE() << Log::no_space << "  Plugin [" << plugin.get_plugin_name() << "] ("
                           << plugin.get_integrator_name() << ")";
        if (plugin.get_camera_discovery_list().empty() && plugin.get_file_discovery_list().empty())
            continue;
        for (auto &camera_discovery : plugin.get_camera_discovery_list()) {
            if (camera_discovery.is_for_local_camera()) {
                // Check local camera
                if ((flag & 2) == 0)
                    continue;
            } else {
                // Check remote camera
                if ((flag & 1) == 0)
                    continue;
            }
            auto list_serial = camera_discovery.list();
            auto log = MV_HAL_LOG_TRACE() << Log::no_endline << "    Camera discovery" << camera_discovery.get_name();
            if (list_serial.empty()) {
                log << "does not recognize any device" << std::endl;
            } else {
                log << "recognizes:" << std::endl;
                for (auto serial : list_serial) {
                    ret.push_back(get_full_serial(plugin.get_integrator_name(), plugin.get_plugin_name(), serial));
                    MV_HAL_LOG_TRACE() << "      Serial:" << serial;
                }
            }
        }
    }

    return ret;
}

DeviceDiscovery::SystemList list_systems_camera(CameraType flag = ANY) {
    DeviceDiscovery::SystemList ret;

    MV_HAL_LOG_TRACE() << "Listing cameras of" << CameraTypeLabels[flag - 1] << "type";

    for (auto &plugin : get_plugins()) {
        MV_HAL_LOG_TRACE() << Log::no_space << "  Plugin [" << plugin.get_plugin_name() << "] ("
                           << plugin.get_integrator_name() << ")";
        bool has_serial = false;
        for (auto &camera_discovery : plugin.get_camera_discovery_list()) {
            if (camera_discovery.is_for_local_camera()) {
                // Check local camera
                if ((flag & 2) == 0)
                    continue;
            } else {
                // Check remote camera
                if ((flag & 1) == 0)
                    continue;
            }

            CameraDiscovery::SystemList list_systems = camera_discovery.list_available_sources();
            auto log = MV_HAL_LOG_TRACE() << Log::no_endline << "    Camera discovery" << camera_discovery.get_name();
            if (list_systems.empty()) {
                log << "does not recognize any device" << std::endl;
            } else {
                log << "recognizes:" << std::endl;
                for (auto plugin_cam_desc : list_systems) {
                    CameraDescription cam_desc(plugin_cam_desc);
                    cam_desc.integrator_name_ = plugin.get_integrator_name();
                    cam_desc.plugin_name_     = plugin.get_plugin_name();
                    ret.push_back(cam_desc);
                    MV_HAL_LOG_TRACE() << "      Serial:" << cam_desc.get_full_serial();
                }
            }
        }
    }

    MV_HAL_LOG_TRACE() << "";

    return ret;
}

DeviceDiscovery::SerialList DeviceDiscovery::list() {
    return list_serial_camera();
}

DeviceDiscovery::SerialList DeviceDiscovery::list_remote() {
    return list_serial_camera(REMOTE);
}

DeviceDiscovery::SerialList DeviceDiscovery::list_local() {
    return list_serial_camera(LOCAL);
}

DeviceDiscovery::SystemList DeviceDiscovery::list_available_sources() {
    return list_systems_camera();
}

DeviceDiscovery::SystemList DeviceDiscovery::list_available_sources_remote() {
    return list_systems_camera(REMOTE);
}

DeviceDiscovery::SystemList DeviceDiscovery::list_available_sources_local() {
    return list_systems_camera(LOCAL);
}

std::unique_ptr<Device> DeviceDiscovery::open(const std::string &serial) {
    DeviceConfig default_config;
    return DeviceDiscovery::open(serial, default_config);
}

std::unique_ptr<Device> DeviceDiscovery::open(const std::string &input_serial, DeviceConfig &config) {
    MV_HAL_LOG_TRACE() << "Opening camera with serial:" << input_serial;

    std::unique_ptr<Device> device;
    std::string integrator_name;
    std::string plugin_name;

    // split name plugin_name:intergrator:serial
    size_t pos             = 0;
    std::string tmp_serial = input_serial;
    std::string delimiter  = ":";
    std::vector<std::string> fields;
    while ((pos = tmp_serial.find(delimiter)) != std::string::npos) {
        auto field = tmp_serial.substr(0, pos);
        fields.push_back(field);
        tmp_serial.erase(0, pos + delimiter.length());
    }
    std::string serial = tmp_serial;

    assert(fields.size() <= 2);
    std::string input_integrator_name;
    std::string input_plugin_name;
    std::string input_common_name;
    if (fields.size() == 1) {
        input_common_name = fields[0];
    }
    if (fields.size() == 2) {
        input_integrator_name = fields[0];
        input_plugin_name     = fields[1];
    }

    for (auto &plugin : get_plugins()) {
        if (device) {
            break;
        }

        plugin_name     = plugin.get_plugin_name();
        integrator_name = plugin.get_integrator_name();
        if ((!input_integrator_name.empty() && input_integrator_name != integrator_name) ||
            (!input_plugin_name.empty() && input_plugin_name != plugin_name)) {
            MV_HAL_LOG_TRACE() << Log::no_space << "  Plugin [" << plugin_name << "] (" << integrator_name
                               << ") does not match the serial";
            continue;
        }
        if (!input_common_name.empty()) {
            if (input_common_name != integrator_name && input_common_name != plugin_name) {
                MV_HAL_LOG_TRACE() << Log::no_space << "  Plugin [" << plugin_name << "] (" << integrator_name
                                   << ") does not match the serial";
                continue;
            }
        }

        MV_HAL_LOG_TRACE() << Log::no_space << "  Plugin [" << plugin.get_plugin_name() << "] ("
                           << plugin.get_integrator_name() << ") matches the serial";
        for (auto &camera_discovery : plugin.get_camera_discovery_list()) {
            if (device) {
                break;
            }
            if (serial == "" && !camera_discovery.is_for_local_camera()) {
                continue;
            }
            MV_HAL_LOG_TRACE() << "    Camera discovery" << camera_discovery.get_name();
            try {
                DeviceBuilder device_builder(
                    std::make_unique<I_HALSoftwareInfo>(plugin.get_hal_info()),
                    std::make_unique<I_PluginSoftwareInfo>(plugin.get_plugin_name(), plugin.get_plugin_info()));
                if (camera_discovery.discover(device_builder, serial, config)) {
                    MV_HAL_LOG_TRACE() << "        -> can open the serial";
                    device = device_builder();
                } else {
                    MV_HAL_LOG_TRACE() << "        -> cannot open the serial";
                }
            } catch (HalException &e) {
                log_plugin_error(plugin, camera_discovery.get_name(), e);
            } catch (const std::exception &e) {
                log_plugin_error(plugin, camera_discovery.get_name(), e);
            } catch (...) { log_plugin_error(plugin, camera_discovery.get_name()); }
        }
    }

    return device;
}

std::unique_ptr<Device> DeviceDiscovery::open_raw_file(const std::string &raw_file) {
    RawFileConfig cfg;
    return open_raw_file(raw_file, cfg);
}

// TODO MV-166: remove this overload
std::unique_ptr<Device> DeviceDiscovery::open_raw_file(const std::string &raw_file, RawFileConfig &file_config) {
    auto ifs = std::make_unique<std::ifstream>(raw_file, std::ios::in | std::ios::binary);
    if (!ifs->good()) {
        throw HalException(HalErrorCode::FailedInitialization, "Unable to open RAW file '" + raw_file + "'");
    }

    std::unique_ptr<Device> device;
    try {
        device             = open_stream(std::move(ifs), file_config);
        auto *event_stream = device->get_facility<I_EventsStream>();
        if (event_stream) {
            event_stream->set_underlying_filename(raw_file);
        }

    } catch (const HalException &e) {
        MV_HAL_LOG_ERROR() << Log::no_space << "While opening RAW file '" << raw_file << "':" << std::endl;
        throw e;
    }

    return device;
}

std::unique_ptr<Device> DeviceDiscovery::open_raw_file(const std::string &raw_file,
                                                       Future::RawFileConfig &file_config) {
    auto ifs = std::make_unique<std::ifstream>(raw_file, std::ios::in | std::ios::binary);
    if (!ifs->good()) {
        throw HalException(HalErrorCode::FailedInitialization, "Unable to open RAW file '" + raw_file + "'");
    }

    RawFileConfig config;
    config.n_events_to_read_ = file_config.n_events_to_read_;
    config.n_read_buffers_   = file_config.n_read_buffers_;
    config.do_time_shifting_ = file_config.do_time_shifting_;
    std::unique_ptr<Device> device;
    try {
        device             = open_stream(std::move(ifs), config);
        auto *event_stream = device->get_facility<I_EventsStream>();
        if (event_stream) {
            event_stream->set_underlying_filename(raw_file);
        }

        auto *future_events_stream = device->get_facility<Future::I_EventsStream>();
        if (future_events_stream) {
            future_events_stream->set_underlying_filename(raw_file);
            if (file_config.build_index_) {
                // We create an additional decicated device that we will use for indexing the RAW file
                // We set build_index_ = false for this device, because it won't be used for seeking, so it does
                // not need to have an index automatically built. Not doing so would create an infinite loop
                // of devices created for the purpose of building the index for the one previously created.
                Future::RawFileConfig cfg;
                cfg.do_time_shifting_    = true;
                cfg.build_index_         = false;
                auto device_for_indexing = open_raw_file(raw_file, cfg);
                if (device_for_indexing) {
                    try {
                        future_events_stream->index(std::move(device_for_indexing));
                    } catch (const HalException &e) {
                        MV_HAL_LOG_TRACE() << "Could not build index for the file. Exception caught:\n" << e.what();
                    }
                }
            }
        }

    } catch (const HalException &e) {
        MV_HAL_LOG_ERROR() << Log::no_space << "While opening RAW file '" << raw_file << "':" << std::endl;
        throw e;
    }

    return device;
}

std::unique_ptr<Device> DeviceDiscovery::open_stream(std::unique_ptr<std::istream> stream,
                                                     RawFileConfig &stream_config) {
    if (!stream) {
        throw HalException(HalErrorCode::FailedInitialization,
                           "Failed to read from input stream: invalid pointer (nullptr)");
    }

    std::unique_ptr<Device> device;

    RawFileHeader header(*stream);

    std::string input_integrator_name = header.get_integrator_name();
    std::string input_plugin_name     = header.get_plugin_name();
    std::string plugin_name, integrator_name;

    MV_HAL_LOG_TRACE() << Log::no_space << "Opening camera from stream, identified as ["
                       << (input_plugin_name.empty() ? "Unknown" : input_plugin_name) << "] ("
                       << (input_integrator_name.empty() ? "Unknown" : input_integrator_name) << ")";

    auto list_plugins = get_plugins();

    for (auto &plugin : list_plugins) {
        if (device) {
            break;
        }

        plugin_name     = plugin.get_plugin_name();
        integrator_name = plugin.get_integrator_name();
        if (input_integrator_name.empty() && input_plugin_name.empty()) {
            if (integrator_name != "Prophesee") {
                // for backward compatibility with our old RAW files where these
                // fields were not yet added in the header
                continue;
            }
        } else if ((!input_integrator_name.empty() && input_integrator_name != integrator_name) ||
                   (!input_plugin_name.empty() && input_plugin_name != plugin_name)) {
            MV_HAL_LOG_TRACE() << Log::no_space << "  Plugin [" << plugin_name << "] (" << integrator_name
                               << ") does not match the header";
            continue;
        }

        MV_HAL_LOG_TRACE() << Log::no_space << "  Plugin [" << plugin.get_plugin_name() << "] ("
                           << plugin.get_integrator_name() << ") matches the header";
        for (auto &file_discovery : plugin.get_file_discovery_list()) {
            if (device) {
                break;
            }
            MV_HAL_LOG_TRACE() << "    File discovery" << file_discovery.get_name();
            try {
                DeviceBuilder device_builder(
                    std::make_unique<I_HALSoftwareInfo>(plugin.get_hal_info()),
                    std::make_unique<I_PluginSoftwareInfo>(plugin.get_plugin_name(), plugin.get_plugin_info()));
                if (file_discovery.discover(device_builder, stream, header, stream_config)) {
                    MV_HAL_LOG_TRACE() << "      -> Can open the file";
                    device = device_builder();
                } else {
                    MV_HAL_LOG_TRACE() << "      -> Cannot open the file";
                }
            } catch (HalException &e) {
                log_plugin_error(plugin, file_discovery.get_name(), e);
            } catch (const std::exception &e) { log_plugin_error(plugin, file_discovery.get_name(), e); } catch (...) {
                log_plugin_error(plugin, file_discovery.get_name());
            }
            if (!device && !stream) {
                // We can get here if the implementation takes ownership of the stream but the output device is
                // null. The requirements (see documentation of the 'discover' method in the FileDiscovery) have
                // not been fulfilled.
                throw HalException(HalErrorCode::FailedInitialization,
                                   "The plugin was expected to be able to read from the input stream, but a "
                                   "null device was constructed.");
            }
        }
    }

    if (device) {
        return device;
    }

    if (!header.empty()) {
        throw HalException(HalErrorCode::FailedInitialization,
                           "No plugin available for input stream. Could not identify source from header:\n" +
                               header.to_string());
    } else {
        throw HalException(HalErrorCode::FailedInitialization,
                           "No plugin available for input stream. Could not identify source without a header.");
    }

    return nullptr;
}

} // namespace Metavision
