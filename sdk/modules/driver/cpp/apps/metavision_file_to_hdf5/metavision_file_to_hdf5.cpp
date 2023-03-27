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

// This application demonstrates how to use Metavision SDK Core pipeline utility and SDK Driver to convert a file
// to HDF5 file.

#include <iostream>
#include <functional>
#include <chrono>
#include <cstdio>
#include <regex>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/core/pipeline/stage.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>
#include <metavision/sdk/driver/hdf5_event_file_writer.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string in_file_path;

    const std::string program_desc("Application to convert a file to HDF5 file.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i", po::value<std::string>(&in_file_path)->required(), "Path to input file.")
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
    //  1 (Write CD)                     2 (Write Ext Trigger)
    //

    // 0) Stage producing events from a camera
    Metavision::Camera camera;
    try {
        camera = Metavision::Camera::from_file(in_file_path, Metavision::FileConfigHints().real_time_playback(false));
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }
    auto &cam_stage = p.add_stage(std::make_unique<Metavision::CameraStage>(std::move(camera)));
    // Gets the wrapped camera from the stage to extract sensor's resolution
    Metavision::Camera &cam = cam_stage.camera();

    // Get the output filename and build a HDF5 file writer
    std::string out_hdf5_file_path = boost::filesystem::extension(in_file_path) != "" ?
                                         std::regex_replace(in_file_path, std::regex("\\.[^.]*$"), ".hdf5") :
                                         in_file_path + ".hdf5";
    if (in_file_path == out_hdf5_file_path) {
        MV_LOG_ERROR() << "Error : input file has HDF5 extension. Please, make sure the input file does not have .hdf5 "
                          "extension or rename it.";
        return 1;
    }

    Metavision::HDF5EventFileWriter hdf5_writer(out_hdf5_file_path);
    hdf5_writer.add_metadata_map_from_camera(cam);

    // 1) Stage that will write CD events to a HDF5 file
    auto cd_stage_ptr = std::make_unique<Metavision::Stage>();
    cd_stage_ptr->set_consuming_callback([&hdf5_writer](const boost::any &data) {
        try {
            auto buffer = boost::any_cast<Metavision::Stage::EventBufferPtr>(data);
            hdf5_writer.add_events(buffer->data(), buffer->data() + buffer->size());
        } catch (boost::bad_any_cast &) {}
    });
    p.add_stage(std::move(cd_stage_ptr), cam_stage);

    // 2) Stage that will write external triggers events to a HDF5 file
    try {
        cam.ext_trigger().add_callback(
            [&cam_stage](const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
                cam_stage.add_ext_trigger_events(begin, end);
            });

        auto ext_trigger_stage_ptr = std::make_unique<Metavision::Stage>();
        ext_trigger_stage_ptr->set_consuming_callback([&hdf5_writer](const boost::any &data) {
            try {
                auto buffer = boost::any_cast<Metavision::CameraStage::EventTriggerBufferPtr>(data);
                hdf5_writer.add_events(buffer->data(), buffer->data() + buffer->size());
            } catch (boost::bad_any_cast &) {}
        });
        p.add_stage(std::move(ext_trigger_stage_ptr), cam_stage);
    } catch (Metavision::CameraException &) {}

    // Run the pipeline step by step to give a visual feedback about the progression
    using namespace std::chrono_literals;
    auto log = MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline;
    const std::string message("Writing HDF5 file...");
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

    hdf5_writer.close();

    MV_LOG_INFO() << "\rWrote HDF5 file " << out_hdf5_file_path;

    return 0;
}
