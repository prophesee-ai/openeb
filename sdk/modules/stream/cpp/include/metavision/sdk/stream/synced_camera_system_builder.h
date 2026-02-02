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

#ifndef METAVISION_SDK_DRIVER_SYNCED_CAMERA_SYSTEM_BUILDER_H
#define METAVISION_SDK_DRIVER_SYNCED_CAMERA_SYSTEM_BUILDER_H

#include <variant>

#include "metavision/sdk/stream/synced_camera_system_factory.h"
#include "metavision/sdk/stream/file_config_hints.h"

namespace Metavision {

/// @brief Builder class to create a synced camera system
///
/// Depending on the provided parameters, the builder will create a synced camera system with live cameras or with
/// files.
class SyncedCameraSystemBuilder {
public:
    /// @brief Adds live camera parameters to the builder
    ///
    /// The first provided parameters will be used as the master camera parameters
    /// @param parameters Live camera parameters
    /// @return Reference to the builder
    SyncedCameraSystemBuilder &
        add_live_camera_parameters(const SyncedCameraSystemFactory::LiveCameraParameters &parameters);

    /// @brief Sets whether the system should record the live events or not
    /// @param record True to record the events, false otherwise
    /// @return Reference to the builder
    SyncedCameraSystemBuilder &set_record(bool record);

    /// @brief Sets the directory where the events will be recorded
    ///
    /// Setting this option will automatically enable recording
    /// @param record_dir Directory where the events will be recorded
    /// @return Reference to the builder
    SyncedCameraSystemBuilder &set_record_dir(const std::filesystem::path &record_dir);

    /// @brief Adds a path to a record file
    ///
    /// The first provided path will be used as the master camera file path
    /// @param record_path Path to a record file
    /// @return Reference to the builder
    SyncedCameraSystemBuilder &add_record_path(const std::filesystem::path &record_path);

    /// @brief Sets the file config hints
    ///
    /// The hints will be used for all the records
    /// @param file_config_hints File config hints
    /// @return Reference to the builder
    SyncedCameraSystemBuilder &set_file_config_hints(const FileConfigHints &file_config_hints);

    /// @brief Builds the synced camera system
    /// @return Tuple containing the master camera and the slave cameras
    /// @throw std::runtime_error in case:
    ///     - No camera parameters provided
    ///     - Mixing live and offline parameters
    ///     - Less than two live cameras provided
    ///     - Less than two records provided
    std::tuple<Camera, std::vector<Camera>> build();

private:
    std::vector<SyncedCameraSystemFactory::LiveCameraParameters> live_parameters_;
    bool record_;
    std::optional<std::filesystem::path> record_dir_;
    std::vector<std::filesystem::path> file_paths_;
    FileConfigHints file_config_hints_;
};

} // namespace Metavision

#endif // METAVISION_SDK_DRIVER_SYNCED_CAMERA_SYSTEM_BUILDER_H
