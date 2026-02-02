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

#ifndef METAVISION_SDK_DRIVER_SYNCED_CAMERA_SYSTEM_FACTORY_H
#define METAVISION_SDK_DRIVER_SYNCED_CAMERA_SYSTEM_FACTORY_H

#include <string>
#include <optional>
#include <filesystem>

#include "metavision/hal/utils/device_config.h"
#include "metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h"

namespace Metavision {
class Camera;

class SyncedCameraSystemFactory {
public:
    /// @brief Parameters to create a live camera
    struct LiveCameraParameters {
        std::string serial_number;                               ///< Serial number of the camera
        DeviceConfig device_config;                              ///< Device configuration
        std::optional<std::filesystem::path> settings_file_path; ///< Optional path to a camera settings file
    };

    /// @brief Parameters to create a synced live camera system
    struct LiveParameters {
        LiveCameraParameters master_parameters;             ///< Parameters for the master camera
        std::vector<LiveCameraParameters> slave_parameters; ///< Parameters for the slave cameras
        bool record;                                        ///< True to record the events, false otherwise
        std::optional<std::filesystem::path> record_dir;    ///< Optional directory where the events will be recorded
    };

    /// @brief Parameters to create a synced offline camera system
    struct OfflineParameters {
        std::filesystem::path master_file_path;              ///< Path to the master record
        std::vector<std::filesystem::path> slave_file_paths; ///< Paths to the slave records
        FileConfigHints file_config_hints;                   ///< Hints to configure the file reading
    };

    /// @brief Builds a synced camera system with live cameras
    /// @param parameters Parameters to create the synced camera system
    /// @return A tuple with the master camera and the slave cameras
    static std::tuple<Camera, std::vector<Camera>> build(const LiveParameters &parameters);

    /// @brief Builds a synced camera system with offline cameras
    /// @param parameters Parameters to create the synced camera system
    /// @return A tuple with the master camera and the slave cameras
    static std::tuple<Camera, std::vector<Camera>> build(const OfflineParameters &parameters);
};
} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_SYNCED_CAMERA_SYSTEM_BUILDER_H
