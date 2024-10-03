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

// This code demonstrates how to use the Metavision CameraStreamSlicer to slice the events from a camera or a file into
// slices of a fixed number of events or a fixed duration.

#include <boost/assign/list_of.hpp>
#include <boost/bimap.hpp>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <metavision/sdk/core/algorithms/base_frame_generation_algorithm.h>
#include <metavision/sdk/stream/camera_stream_slicer.h>

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
    std::string record_path;
    std::string serial_number;
    SlicingCondition slicing_condition;
};

std::optional<Config> parse_command_line(int argc, char *argv[]) {
    namespace po = boost::program_options;

    Metavision::timestamp delta_ts;
    size_t delta_n_events;
    SlicingMode slicing_mode;

    const std::string program_desc("Code sample showing how to use the Metavision Slicer");

    Config config;
    po::options_description options_desc;
    po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i", po::value<std::string>(&config.record_path), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("camera-serial-number,s", po::value<std::string>(&config.serial_number), "Serial number of the camera to be used")
        ;
    // clang-format on

    po::options_description slicing_options("Slicing options");
    // clang-format off
    slicing_options.add_options()
        ("slicing-mode,m", po::value<SlicingMode>(&slicing_mode)->default_value(SlicingMode::N_US), "Slicing mode (i.e. N_EVENTS, N_US, MIXED)")
        ("delta-ts,t", po::value<Metavision::timestamp>(&delta_ts)->default_value(10000), "Slice duration in us")
        ("delta-n-events,n", po::value<size_t>(&delta_n_events)->default_value(10000), "Number of events in a slice")
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

int main(int argc, char *argv[]) {
    const auto config = parse_command_line(argc, argv);
    if (!config)
        return 1;

    /// [CAMERA_INIT_BEGIN]
    Metavision::Camera camera;

    if (!config->record_path.empty()) {
        camera = Metavision::Camera::from_file(config->record_path);
    } else if (!config->serial_number.empty()) {
        camera = Metavision::Camera::from_serial(config->serial_number);
    } else {
        camera = Metavision::Camera::from_first_available();
    }
    /// [CAMERA_INIT_END]

    const auto &geometry = camera.geometry();
    const auto width     = geometry.get_width();
    const auto height    = geometry.get_height();

    cv::Mat frame_8uc3(height, width, CV_8UC3);

    /// [SLICER_INIT_BEGIN]
    Metavision::CameraStreamSlicer slicer(std::move(camera), config->slicing_condition);
    /// [SLICER_INIT_END]

    /// [SLICER_LOOP_BEGIN]
    for (const auto &slice : slicer) {
        MV_LOG_INFO() << "ts:" << slice.t << "new slice of" << slice.n_events << "events";

        frame_8uc3.create(height, width, CV_8UC3);
        Metavision::BaseFrameGenerationAlgorithm::generate_frame_from_events(slice.events->cbegin(),
                                                                             slice.events->cend(), frame_8uc3);

        for (const auto &t : *slice.triggers) {
            MV_LOG_INFO() << "ts:" << t.t << "new external trigger with polarity" << t.p;
        }

        cv::imshow("Camera stream slicer", frame_8uc3);
        const auto cmd = cv::waitKey(1);
        if (cmd == 'q')
            break;
    }
    /// [SLICER_LOOP_END]

    return 0;
}