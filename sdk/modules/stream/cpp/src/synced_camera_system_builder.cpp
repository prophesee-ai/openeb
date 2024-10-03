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

#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/synced_camera_system_builder.h"

namespace Metavision {

SyncedCameraSystemBuilder &SyncedCameraSystemBuilder::add_live_camera_parameters(
    const SyncedCameraSystemFactory::LiveCameraParameters &parameters) {
    live_parameters_.emplace_back(parameters);

    return *this;
}

SyncedCameraSystemBuilder &SyncedCameraSystemBuilder::set_record(bool record) {
    record_ = record;
    return *this;
}

SyncedCameraSystemBuilder &SyncedCameraSystemBuilder::set_record_dir(const std::filesystem::path &record_dir) {
    record_dir_ = record_dir;
    return *this;
}

SyncedCameraSystemBuilder &SyncedCameraSystemBuilder::add_record_path(const std::filesystem::path &record_path) {
    file_paths_.emplace_back(record_path);
    return *this;
}

SyncedCameraSystemBuilder &SyncedCameraSystemBuilder::set_file_config_hints(const FileConfigHints &file_config_hints) {
    file_config_hints_ = file_config_hints;
    return *this;
}

std::tuple<Camera, std::vector<Camera>> SyncedCameraSystemBuilder::build() {
    if (live_parameters_.empty() && file_paths_.empty()) {
        throw std::runtime_error("No camera parameters provided");
    }

    if (!live_parameters_.empty() && !file_paths_.empty()) {
        throw std::runtime_error("Cannot mix live and offline parameters");
    }

    if (live_parameters_.size() == 1) {
        throw std::runtime_error("At least two live cameras are required");
    }

    if (file_paths_.size() == 1) {
        throw std::runtime_error("At least two files are required for offline mode");
    }

    if (!live_parameters_.empty()) {
        SyncedCameraSystemFactory::LiveParameters parameters;
        parameters.master_parameters = live_parameters_.front();
        parameters.slave_parameters.assign(live_parameters_.begin() + 1, live_parameters_.end());
        parameters.record     = record_;
        parameters.record_dir = record_dir_;

        return SyncedCameraSystemFactory::build(parameters);
    }

    SyncedCameraSystemFactory::OfflineParameters parameters;
    parameters.master_file_path = file_paths_.front();
    parameters.slave_file_paths.assign(file_paths_.begin() + 1, file_paths_.end());
    parameters.file_config_hints = file_config_hints_;

    return SyncedCameraSystemFactory::build(parameters);
}
} // namespace Metavision