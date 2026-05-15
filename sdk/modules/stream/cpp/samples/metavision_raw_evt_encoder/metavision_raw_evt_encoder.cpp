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

// This application demonstrates how to use Metavision SDK Stream module to decode an event recording, process it and
// encode it back to RAW EVT2 format.

#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/algorithms/flip_y_algorithm.h>
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/stream/raw_evt2_event_file_writer.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::filesystem::path in_path;
    std::filesystem::path out_path;
    bool encode_triggers                    = false;
    Metavision::timestamp max_event_latency = -1;

    const std::string program_desc("Sample application to process a decoded event stream and encode it to RAW EVT2.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-path,i", po::value<std::filesystem::path>(&in_path)->required(), "Path to input event file.")
        ("output-path,o", po::value<std::filesystem::path>(&out_path)->default_value(""), "Path to output file. If not specified, will use a modified version of the input path.")
        ("encode-triggers", po::bool_switch(&encode_triggers), "Flag to activate encoding of external trigger events.")
        ("max-event-latency", po::value<Metavision::timestamp>(&max_event_latency)->default_value(-1), "Maximum latency in camera time for the reception of events, infinite by default.")
    ;
    // clang-format on

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    try {
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    // Get the output filename
    if (out_path.empty()) {
        out_path = in_path.parent_path() / in_path.stem();
        out_path.concat("_evt_encoded.raw");
    }

    // Create the camera
    Metavision::Camera camera = Metavision::Camera::from_file(in_path.string());
    const auto width          = camera.geometry().get_width();
    const auto height         = camera.geometry().get_height();

    // Instantiate the pipeline for processing decoded events, here a simple flip y algorithm
    Metavision::FlipYAlgorithm yflipper(height - 1);

    // Instantiate the RAW encoder
    Metavision::RAWEvt2EventFileWriter writer(width, height, out_path.string(), encode_triggers, {}, max_event_latency);

    // Setup console feedback to be provided on processing progression
    using namespace std::chrono_literals;
    auto log = MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline;
    log << "Writing to " << out_path << "\n";
    const std::string message("Encoding RAW file...");
    log << message << std::flush;
    int dots                   = 0;
    auto last_time             = std::chrono::high_resolution_clock::now();
    auto progress_feedback_fct = [&]() {
        const auto time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(time - last_time) > 500ms) {
            last_time = time;
            log << "\r" << message.substr(0, message.size() - 3 + dots) + std::string("   ").substr(0, 3 - dots)
                << std::flush;
            dots = (dots + 1) % 4;
        }
    };

    // Define the callback to process the events
    std::vector<Metavision::EventCD> events;
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        events.clear();
        yflipper.process_events(begin, end, std::back_inserter(events));
        writer.add_events(events.data(), events.data() + events.size());
        progress_feedback_fct();
    });
    if (encode_triggers) {
        camera.ext_trigger().add_callback(
            [&](const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
                writer.add_events(begin, end);
            });
    }

    // Start the camera and the processing/encoding
    camera.start();
    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };
    log << "\rDone!                    " << std::endl;
    camera.stop();

    return 0;
}
