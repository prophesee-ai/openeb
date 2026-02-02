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

#include <metavision/hal/utils/camera_discovery.h>
#include <metavision/hal/utils/device_config.h>


/// @brief Discovers connected devices
///
/// This class is the implementation of HAL's class @ref Metavision::CameraDiscovery
class SampleCameraDiscovery : public Metavision::CameraDiscovery {
public:
    Metavision::CameraDiscovery::SerialList list() override final;
    Metavision::CameraDiscovery::SystemList list_available_sources() override final;
    bool discover(Metavision::DeviceBuilder &device_builder, const std::string &serial,
                  const Metavision::DeviceConfig &config) override;
    bool is_for_local_camera() const override final;
};

#endif // METAVISION_HAL_SAMPLE_CAMERA_DISCOVERY_H
