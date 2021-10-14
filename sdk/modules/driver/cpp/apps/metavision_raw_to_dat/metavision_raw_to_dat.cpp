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

// This application demonstrates how to use Metavision SDK Core pipeline utility and SDK Driver to convert RAW file
// to DAT file.

#include <iostream>
#include <functional>
#include <chrono>
#include <cstdio>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/core/pipeline/stream_logging_stage.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string in_raw_file_path;

    const std::string program_desc("Application to convert RAW file to DAT file.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i", po::value<std::string>(&in_raw_file_path)->required(), "Path to input RAW file.")
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

    // A pipeline for which all added stages will automatically be run in their own processing threads (if applicable)
    Metavision::Pipeline p(true);

    /// Pipeline
    //                  0 (Camera)
    //                  |
    //                  v
    //                  |
    //  |----------<--------->-----------|
    //  |                                |
    //  v                                v
    //  |                                |
    //  1 (Log CD)                       2 (Log Ext Trigger)
    //

    // 0) Stage producing events from a camera
    Metavision::Camera camera;
    try {
        camera = Metavision::Camera::from_file(in_raw_file_path, false);
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }
    auto &cam_stage = p.add_stage(std::make_unique<Metavision::CameraStage>(std::move(camera)));
    // Gets the wrapped camera from the stage to extract sensor's resolution
    Metavision::Camera &cam     = cam_stage.camera();
    const unsigned short width  = cam.geometry().width();
    const unsigned short height = cam.geometry().height();

    // Get the base of the input filename and the path
    std::string output_base = in_raw_file_path.substr(0, in_raw_file_path.find_last_of(".raw") - 3);

    // 1) Stage that will write CD events to a DAT file
    std::string cd_filename(output_base + "_cd.dat");
    p.add_stage(std::make_unique<Metavision::StreamLoggingStage<Metavision::EventCD>>(cd_filename, width, height),
                cam_stage);

    // 2) Stage that will write external triggers events to a DAT file
    std::string ext_trigger_filename(output_base + "_trigger.dat");
    bool has_ext_trigger = false;
    try {
        cam.ext_trigger().add_callback([&has_ext_trigger, &cam_stage](const Metavision::EventExtTrigger *begin,
                                                                      const Metavision::EventExtTrigger *end) {
            has_ext_trigger = true;
            cam_stage.add_ext_trigger_events(begin, end);
        });
        // if no external triggers events are available in the recording, then an exception will be thrown and
        // the stage will never be added
        p.add_stage(std::make_unique<Metavision::StreamLoggingStage<Metavision::EventExtTrigger>>(ext_trigger_filename,
                                                                                                  width, height),
                    cam_stage);
    } catch (Metavision::CameraException &) {}

    // Run the pipeline step by step to give a visual feedback about the progression
    using namespace std::chrono_literals;
    auto log = MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline;
    const std::string message("Writing DAT file...");
    int dots       = 0;
    auto last_time = std::chrono::high_resolution_clock::now();

    while (p.step()) {
        const auto time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(time - last_time) > 500ms) {
            last_time = time;
            log << "\r" << message.substr(0, message.size() - 3 + dots) + std::string("   ").substr(0, 3 - dots)
                << std::flush;
            dots = (dots + 1) % 4;
        }
    }

    MV_LOG_INFO() << "\rWrote CD DAT file " << cd_filename;
    if (has_ext_trigger) {
        MV_LOG_INFO() << std::endl << "Wrote external triggers DAT file " << ext_trigger_filename;
    } else {
        std::remove(ext_trigger_filename.c_str());
    }

    return 0;
}
