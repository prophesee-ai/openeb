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

// Demonstrates using Metavision SyncedCameraStreamsSlicer to slice events from a master and slave cameras system into
// fixed slices (i.e. number of events or duration)

#include <fstream>
#include <filesystem>

#include <boost/assign/list_of.hpp>
#include <boost/bimap.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <metavision/sdk/core/algorithms/base_frame_generation_algorithm.h>
#include <metavision/sdk/stream/synced_camera_streams_slicer.h>
#include <metavision/sdk/stream/synced_camera_system_builder.h>
#include <metavision/sdk/stream/synced_camera_system_factory.h>

using SlicingMode      = Metavision::EventBufferReslicerAlgorithm::ConditionType;
using SlicingCondition = Metavision::EventBufferReslicerAlgorithm::Condition;

namespace std {
using Bimap = boost::bimap<SlicingMode, std::string>;

// clang-format off
const Bimap kSlicingModeToStr = boost::assign::list_of<Bimap::relation>
    (SlicingMode::IDENTITY, "IDENTITY")
    (SlicingMode::N_EVENTS, "N_EVENTS")
    (SlicingMode::N_US, "N_US")
    (SlicingMode::MIXED, "MIXED");
// clang-format on

std::istream &operator>>(std::istream &is, SlicingMode &mode) {
    std::string s;
    is >> s;
    auto it = kSlicingModeToStr.right.find(s);
    if (it == kSlicingModeToStr.right.end())
        throw std::runtime_error("Failed to convert string to slicing mode");
    mode = it->second;
    return is;
}

std::ostream &operator<<(std::ostream &os, const SlicingMode &mode) {
    auto it = kSlicingModeToStr.left.find(mode);
    if (it == kSlicingModeToStr.left.end())
        throw std::runtime_error("Failed to convert slicing mode to string");
    os << it->second;
    return os;
}
} // namespace std

struct Config {
    std::vector<std::string> record_paths;
    std::vector<std::string> serial_numbers;
    bool real_time_playback;
    bool record;
    std::string record_dir;
    std::string settings_dir;

    SlicingCondition slicing_condition;
};

std::optional<Config> parse_command_line(int argc, char *argv[]) {
    namespace po = boost::program_options;

    Metavision::timestamp delta_ts;
    size_t delta_n_events;
    SlicingMode slicing_mode;

    const std::string program_desc(
        "Code sample showing how to use the Metavision SyncedCameraStreamsSlicer to slice events "
        "from a master and slave cameras system into fixed slices");

    Config config;
    po::options_description options_desc;
    po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-event-files,i", po::value<std::vector<std::string>>(&config.record_paths)->multitoken(), "Paths to input event file (RAW or HDF5, first is master). If not specified, the camera live streams are used.")
        ("camera-serial-numbers,s", po::value<std::vector<std::string>>(&config.serial_numbers)->multitoken(), "Serial numbers of the cameras to be used (first is master)")
        ("real-time-playback,r", po::value<bool>(&config.real_time_playback)->default_value(true), "Flag to play records at recording speed")
        ("record", po::value<bool>(&config.record)->default_value(false), "Flag to record the streams")
        ("record-path", po::value<std::string>(&config.record_dir)->default_value(""), "Path to save the recorded streams")
        ("settings-path", po::value<std::string>(&config.settings_dir)->default_value(""), "Path from where to load the settings file for each live camera")
;
    // clang-format on

    po::options_description slicing_options("Slicing options");
    // clang-format off
    slicing_options.add_options()
        ("slicing-mode,m", po::value<SlicingMode>(&slicing_mode)->default_value(SlicingMode::N_US), "Slicing mode (i.e. N_EVENTS, N_US, N_MIXED")
        ("delta-ts,t", po::value<Metavision::timestamp>(&delta_ts)->default_value(10000), "Slice duration in us")
        ("delta-n-events,n", po::value<size_t>(&delta_n_events)->default_value(100000), "Number of events in a slice")
        ;
    // clang-format on

    options_desc.add(base_options).add(slicing_options);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return std::nullopt;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return std::nullopt;
    }

    if (config.record_paths.empty() && config.serial_numbers.empty()) {
        MV_LOG_ERROR() << "At least one input event file or camera serial number must be provided";
        return std::nullopt;
    }

    switch (slicing_mode) {
    case SlicingMode::IDENTITY:
        config.slicing_condition = SlicingCondition::make_identity();
        break;

    case SlicingMode::N_EVENTS:
        config.slicing_condition = SlicingCondition::make_n_events(delta_n_events);
        break;

    case SlicingMode::N_US:
        config.slicing_condition = SlicingCondition::make_n_us(delta_ts);
        break;

    case SlicingMode::MIXED:
        config.slicing_condition = SlicingCondition::make_mixed(delta_ts, delta_n_events);
        break;
    }

    return config;
}

Metavision::SyncedCameraStreamsSlicer build_slicer(const Config &config) {
    /// [BUILD_CAMERA_SYSTEM_BEGIN]
    Metavision::SyncedCameraSystemBuilder builder;

    const auto get_settings_file_path = [&config](const std::string &serial) -> std::optional<std::filesystem::path> {
        namespace fs = std::filesystem;

        const auto settings_file_path = fs::path(config.settings_dir) / (serial + ".json");
        if (!fs::exists(settings_file_path)) {
            return std::nullopt;
        }

        return settings_file_path;
    };

    for (const auto &serial_number : config.serial_numbers) {
        builder.add_live_camera_parameters({serial_number, {}, get_settings_file_path(serial_number)});
    }

    builder.set_record(config.record);
    builder.set_record_dir(config.record_dir);

    for (const auto &path : config.record_paths) {
        builder.add_record_path(path);
    }

    builder.set_file_config_hints(Metavision::FileConfigHints{}.real_time_playback(config.real_time_playback));

    auto &&[master, slaves] = builder.build();
    /// [BUILD_CAMERA_SYSTEM_END]

    return {std::move(master), std::move(slaves), config.slicing_condition};
}

int main(int argc, char *argv[]) {
    const auto config = parse_command_line(argc, argv);
    if (!config)
        return 1;

    auto slicer = build_slicer(*config);

    std::vector<cv::Mat> slice_frames;

    const auto &master_geometry = slicer.master().geometry();
    slice_frames.emplace_back(master_geometry.get_height(), master_geometry.get_width(), CV_8UC3);

    for (size_t i = 0; i < slicer.slaves_count(); ++i) {
        const auto &slave_geometry = slicer.slave(i).geometry();
        slice_frames.emplace_back(slave_geometry.get_height(), slave_geometry.get_width(), CV_8UC3);
    }

    /// [SLICER_LOOP_BEGIN]
    for (const auto &slice : slicer) {
        for (auto &frame : slice_frames) {
            frame.setTo(0);
        }

        MV_LOG_INFO() << "MASTER ts: " << slice.t << " " << slice.n_events << " [" << slice.master_events->front().t
                      << ", " << slice.master_events->back().t << "]";

        Metavision::BaseFrameGenerationAlgorithm::generate_frame_from_events(
            slice.master_events->cbegin(), slice.master_events->cend(), slice_frames[0]);

        cv::imshow("Master slice", slice_frames[0]);

        for (size_t i = 0; i < slice.slave_events.size(); ++i) {
            const auto &slave_slice = slice.slave_events[i];

            MV_LOG_INFO() << "SLAVE " << i + 1 << " ts: " << slice.t << " " << slave_slice->size() << " ["
                          << slave_slice->front().t << ", " << slave_slice->back().t << "]";

            Metavision::BaseFrameGenerationAlgorithm::generate_frame_from_events(
                slave_slice->cbegin(), slave_slice->cend(), slice_frames[i + 1]);

            cv::imshow("Slave slice " + std::to_string(i + 1), slice_frames[i + 1]);
        }

        const auto key = cv::waitKey(1);
        if (key == 'q')
            break;
    }
    /// [SLICER_LOOP_END]

    return 0;
}