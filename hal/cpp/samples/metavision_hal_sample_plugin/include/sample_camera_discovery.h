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

#ifndef METAVISION_HAL_SAMPLE_CAMERA_DISCOVERY_H
#define METAVISION_HAL_SAMPLE_CAMERA_DISCOVERY_H

#include <string>

#include <metavision/hal/utils/camera_discovery.h>
#include <metavision/hal/utils/device_config.h>

/// @brief Discovers connected devices
///
/// This class is the implementation of HAL's class @ref Metavision::CameraDiscovery
class SampleCameraDiscovery : public Metavision::CameraDiscovery {
public:
    /// @brief Lists serial number of available connected devices
    ///
    /// @return List of serial number
    Metavision::CameraDiscovery::SerialList list() override final;

    /// @brief Lists system information about available connected devices
    ///
    /// @return List of system information
    Metavision::CameraDiscovery::SystemList list_available_sources() override final;

    /// @brief Discovers a device and initializes a corresponding @ref DeviceBuilder
    /// @param device_builder Device builder to configure so that it can build a @ref Device from the parameters
    /// @param serial Serial number of the camera to open. If it is an empty string, the first available camera will be
    /// opened
    /// @param config Configuration of camera creation
    /// @return true if a device builder could be discovered from the parameters
    bool discover(Metavision::DeviceBuilder &device_builder, const std::string &serial,
                  const Metavision::DeviceConfig &config) override;

    /// @brief Tells if this CameraDiscovery detects camera locally plugged (USB/MIPI/...) as opposed to remote
    /// camera running on another system
    /// @return true if this camera discovery connect local devices
    bool is_for_local_camera() const override final;
};

#endif // METAVISION_HAL_SAMPLE_CAMERA_DISCOVERY_H
