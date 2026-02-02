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

#include <filesystem>
#include <iomanip>

#include "metavision/hal/facilities/i_camera_synchronization.h"
#include "metavision/sdk/stream/camera.h"
#include "metavision/sdk/stream/synced_camera_system_factory.h"

namespace Metavision {

std::tuple<Camera, std::vector<Camera>> SyncedCameraSystemFactory::build(const LiveParameters &parameters) {
    const auto &master_parameters = parameters.master_parameters;
    auto master_camera = Camera::from_serial(master_parameters.serial_number, master_parameters.device_config);
    if (master_parameters.settings_file_path) {
        const bool success = master_camera.load(*master_parameters.settings_file_path);
        if (!success) {
            throw std::runtime_error("Failed to load master camera config file: " +
                                     master_parameters.settings_file_path->string());
        }
    }

    // Force the master mode
    auto *i_master_sync = master_camera.get_device().get_facility<Metavision::I_CameraSynchronization>();
    const bool success  = i_master_sync->set_mode_master();

    if (!success) {
        throw std::runtime_error("Failed to set master mode for camera with serial number: " +
                                 master_parameters.serial_number);
    }

    std::vector<Camera> slave_cameras;
    for (const auto &settings : parameters.slave_parameters) {
        slave_cameras.push_back(Camera::from_serial(settings.serial_number, settings.device_config));
        auto &slave_camera = slave_cameras.back();
        if (settings.settings_file_path) {
            const bool success = slave_camera.load(*settings.settings_file_path);
            if (!success) {
                throw std::runtime_error("Failed to load slave camera config file: " +
                                         settings.settings_file_path->string());
            }
        }

        // Force the slave mode
        auto *i_slave_sync = slave_camera.get_device().get_facility<Metavision::I_CameraSynchronization>();
        const bool success = i_slave_sync->set_mode_slave();

        if (!success) {
            throw std::runtime_error("Failed to set slave mode for camera with serial number: " +
                                     settings.serial_number);
        }
    }

    if (parameters.record) {
        namespace fs                      = std::filesystem;
        const fs::path recording_dir_path = parameters.record_dir ? fs::path(*parameters.record_dir) :
                                                                    fs::temp_directory_path() / "metavision/recordings";
        if (!fs::exists(recording_dir_path)) {
            fs::create_directories(recording_dir_path);
        }

        const auto current_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        const auto *local_time  = std::localtime(&current_time);
        std::ostringstream date_str;
        date_str << std::put_time(local_time, "%Y-%m-%d_%H-%M-%S");

        const auto &master_recording_path =
            recording_dir_path / ("recording_" + master_parameters.serial_number + "_" + date_str.str() + ".raw");
        master_camera.start_recording(master_recording_path.string());

        for (size_t i = 0; i < slave_cameras.size(); ++i) {
            const auto &slave_recording_path =
                recording_dir_path /
                ("recording_" + parameters.slave_parameters[i].serial_number + "_" + date_str.str() + ".raw");
            slave_cameras[i].start_recording(slave_recording_path.string());
        }
    }

    return {std::move(master_camera), std::move(slave_cameras)};
}

std::tuple<Camera, std::vector<Camera>> SyncedCameraSystemFactory::build(const OfflineParameters &settings) {
    auto master_camera = Metavision::Camera::from_file(settings.master_file_path, settings.file_config_hints);
    std::vector<Metavision::Camera> slave_cameras;
    for (const auto &slave_file : settings.slave_file_paths) {
        slave_cameras.emplace_back(Metavision::Camera::from_file(slave_file, settings.file_config_hints));
    }

    return {std::move(master_camera), std::move(slave_cameras)};
}
} // namespace Metavision