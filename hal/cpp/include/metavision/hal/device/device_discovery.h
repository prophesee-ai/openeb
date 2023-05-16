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

#ifndef METAVISION_HAL_DEVICE_DISCOVERY_H
#define METAVISION_HAL_DEVICE_DISCOVERY_H

#include <list>
#include <memory>
#include <string>

#include "metavision/hal/utils/device_config.h"
#include "metavision/hal/utils/raw_file_config.h"

namespace Metavision {

class Device;

/// @brief Types of links
enum ConnectionType {
    MIPI_LINK        = 1,
    USB_LINK         = 2,
    NETWORK_LINK     = 3,
    PROPRIETARY_LINK = 4,
};

/// @brief Struct returned by the plugin with the camera information
struct PluginCameraDescription {
    /// Serial number of the camera
    std::string serial_;

    /// Type of connection used to communicate with the camera
    ConnectionType connection_;

    /// System Identification number
    long system_id_;
};

/// @brief Overloads operator == for class PluginCameraDescription
bool operator==(const PluginCameraDescription &lhs, const PluginCameraDescription &rhs);

/// @brief Overloads operator != for class PluginCameraDescription
bool operator!=(const PluginCameraDescription &lhs, const PluginCameraDescription &rhs);

/// @brief Struct to store camera information
struct CameraDescription : public PluginCameraDescription {
    /// @brief Copy constructor from base class
    CameraDescription(const PluginCameraDescription &desc) : PluginCameraDescription(desc) {}

    /// @brief Integrator's name
    std::string integrator_name_;

    /// @brief Plugin's name
    std::string plugin_name_;

    /// @brief Returns a string identifying uniquely each camera in the form "integrator:plugin:serial"
    std::string get_full_serial() const;
};

/// @brief Overloads operator == for class CameraDescription
bool operator==(const CameraDescription &lhs, const CameraDescription &rhs);

/// @brief Overloads operator != for class CameraDescription
bool operator!=(const CameraDescription &lhs, const CameraDescription &rhs);

/// @brief Discovery of connected device
class DeviceDiscovery {
public:
    /// @brief Alias to list cameras' serial numbers
    using SerialList = std::list<std::string>;

    /// @brief Alias to list @ref CameraDescription
    using SystemList = std::list<CameraDescription>;

    /// @brief Lists serial numbers of available sources, including remote cameras
    static SerialList list();

    /// @brief Lists serial numbers of local available sources.
    /// @note A camera is considered local if we communicate directly with it.
    static SerialList list_local();

    /// @brief Lists serial numbers of remote available sources.
    /// @note A camera is considered remote if it is part of another SoC and hence communication is indirect.
    static SerialList list_remote();

    /// @brief Lists available sources, including remote cameras
    static SystemList list_available_sources();

    /// @brief Lists only local available sources.
    /// @note A source is considered local if we communicate directly with it.
    static SystemList list_available_sources_local();

    /// @brief Lists only remote available sources.
    /// @note A source is considered remote if it is part of another SoC and hence communication is indirect.
    static SystemList list_available_sources_remote();

    /// @brief Lists DeviceConfig options supported by the camera
    /// @param serial Serial number of the camera which options' should be listed. If it is an empty string, the first
    /// available camera will be considered
    /// @return A map of (key,option) that represent the DeviceConfig options
    static DeviceConfigOptionMap list_device_config_options(const std::string &serial);

    /// @brief Builds a new Device
    /// @param serial Serial number of the camera to open. If it is an empty string, the first available camera will be
    /// opened
    /// @return A new Device
    static std::unique_ptr<Device> open(const std::string &serial);

    /// @brief Builds a new Device
    /// @param serial Serial number of the camera to open. If it is an empty string, the first available camera will be
    /// opened
    /// @param config Configuration used to build the camera
    /// @return A new Device
    static std::unique_ptr<Device> open(const std::string &serial, DeviceConfig &config);

    /// @brief Builds a new Device from file
    /// @param raw_file Path to the file to open
    /// @return A new Device
    static std::unique_ptr<Device> open_raw_file(const std::string &raw_file);

    /// @brief Builds a new Device from file
    /// @param raw_file Path to the file to open
    /// @param file_config Configuration describing how to read the file (see @ref RawFileConfig)
    /// @return A new Device
    static std::unique_ptr<Device> open_raw_file(const std::string &raw_file, RawFileConfig &file_config);

    /// @brief Builds a new Device from a standard input stream
    /// @param stream The input stream to read from. The device takes ownership of the input stream to ensure its
    /// validity during the lifetime of the device object.
    /// @param stream_config Configuration describing how to read the stream (see @ref RawFileConfig)
    /// @return A new Device
    static std::unique_ptr<Device> open_stream(std::unique_ptr<std::istream> stream, RawFileConfig &stream_config);
};

} // namespace Metavision

#endif // METAVISION_HAL_DEVICE_DISCOVERY_H
