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

#ifndef METAVISION_HAL_PLUGIN_H
#define METAVISION_HAL_PLUGIN_H

#include <iterator>
#include <string>
#include <memory>
#include <vector>

#include "metavision/sdk/base/utils/software_info.h"

namespace Metavision {

class CameraDiscovery;
class FileDiscovery;
class PluginLoader;

/// @brief Content of a plugin
class Plugin {
public:
    /// @brief Destructor
    ~Plugin();

    /// @brief Provides the plugin name
    /// @return The name of the plugin
    const std::string &get_plugin_name() const;

    /// @brief Gets the integrator name
    /// @return The name of the camera integrator that the plugin handle
    const std::string &get_integrator_name() const;

    /// @brief Sets the integrator name
    /// @param integrator_name The name of the camera integrator that the plugin handle
    void set_integrator_name(const std::string &integrator_name);

    /// @brief Returns array of object discovering connected devices
    /// @return Object wrapping a list of CameraDiscovery
    class CameraDiscoveryList;
    CameraDiscoveryList get_camera_discovery_list();

    /// @brief Returns an array of object building device from file
    /// @return Object wrapping a list of FileDiscovery
    class FileDiscoveryList;
    FileDiscoveryList get_file_discovery_list();

    /// @brief Appends the camera discovery to the list
    /// @tparam CameraDiscoveryType Type of CameraDiscovery
    /// @param p The camera discovery to add
    /// @return T& A reference to the added camera discovery for convenience
    template<typename CameraDiscoveryType>
    CameraDiscoveryType &add_camera_discovery(std::unique_ptr<CameraDiscoveryType> p) {
        return static_cast<CameraDiscoveryType &>(add_camera_discovery_priv(std::move(p)));
    }

    /// @brief Appends the file discovery to the list
    /// @tparam FileDiscoveryType Type of FileDiscovery
    /// @param p The file discovery to add
    /// @return T& A reference to the added file discovery for convenience
    template<typename FileDiscoveryType>
    FileDiscoveryType &add_file_discovery(std::unique_ptr<FileDiscoveryType> p) {
        return static_cast<FileDiscoveryType &>(add_file_discovery_priv(std::move(p)));
    }

    /// @brief Returns information about the plugin's software
    /// @return Plugin software info
    const SoftwareInfo &get_plugin_info() const;

    /// @brief Sets the plugin software info
    /// @param info Plugin software info
    void set_plugin_info(const SoftwareInfo &info);

    /// @brief Returns information about the hal software with which the plugin was compiled
    /// @return Hal software info
    const SoftwareInfo &get_hal_info() const;

    /// @brief Sets the HAL software info
    /// @param info HAL software info
    void set_hal_info(const SoftwareInfo &info);

private:
    Plugin(const std::string &plugin_name);

    CameraDiscovery &add_camera_discovery_priv(std::unique_ptr<CameraDiscovery> idb);
    FileDiscovery &add_file_discovery_priv(std::unique_ptr<FileDiscovery> idb);

    std::string plugin_name_;
    std::string integrator_name_;
    std::vector<std::unique_ptr<CameraDiscovery>> camera_discovery_list_;
    std::vector<std::unique_ptr<FileDiscovery>> file_discovery_list_;
    std::unique_ptr<SoftwareInfo> plugin_info_;
    std::unique_ptr<SoftwareInfo> hal_info_;

    friend class CameraDiscoveryList;
    friend class FileDiscoveryList;
    friend class PluginLoader;
};

} // namespace Metavision

#include "metavision/hal/plugin/detail/plugin_impl.h"

#endif // METAVISION_HAL_PLUGIN_H
