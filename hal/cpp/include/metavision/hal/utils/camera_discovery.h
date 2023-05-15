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

#ifndef METAVISION_HAL_CAMERA_DISCOVERY_H
#define METAVISION_HAL_CAMERA_DISCOVERY_H

#include <string>
#include <list>
#include <set>
#include <vector>
#include <utility>

#include "metavision/hal/device/device_discovery.h"

namespace Metavision {

class DeviceBuilder;
class DeviceConfig;

/// @brief Discovers connected devices
class CameraDiscovery {
public:
    /// @brief Destructor
    virtual ~CameraDiscovery();

    /// @brief Gets name of Camera Discovery's type
    /// @return Name of Camera Discovery's type
    std::string get_name() const;

    /// @brief Alias to list cameras' serial numbers
    using SerialList = std::list<std::string>;

    /// @brief Alias to list @ref PluginCameraDescription
    using SystemList = std::list<PluginCameraDescription>;

    /// @brief Lists all connected cameras' serial numbers
    virtual SerialList list() = 0;

    /// @brief Lists all @ref PluginCameraDescription of connected cameras
    virtual SystemList list_available_sources() = 0;

    /// @brief Discovers a device and initializes a corresponding @ref DeviceBuilder
    /// @param device_builder Device builder to configure so that it can build a @ref Device from the parameters
    /// @param serial Serial number of the camera to open. If it is an empty string, the first available camera will be
    /// opened
    /// @param config Configuration of camera creation
    /// @return true if a device builder could be discovered from the parameters
    virtual bool discover(DeviceBuilder &device_builder, const std::string &serial, const DeviceConfig &config) = 0;

    /// @brief Tells if this CameraDiscovery detect camera locally plugged (USB/MIPI/...) as opposed to remote
    /// camera running on another system
    virtual bool is_for_local_camera() const;
};

} // namespace Metavision

#endif // METAVISION_HAL_CAMERA_DISCOVERY_H
