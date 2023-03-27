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

// This application demonstrates how to use Metavision SDK Driver to convert a file to DAT file.

#include <iostream>
#include <functional>
#include <chrono>
#include <cstdio>
#include <regex>
#include <thread>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/algorithms/stream_logger_algorithm.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string in_file_path;

    const std::string program_desc("Application to convert a file to DAT file.\n");

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

    Metavision::Camera camera;

    try {
        camera = Metavision::Camera::from_file(in_file_path, Metavision::FileConfigHints().real_time_playback(false));
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }
    const unsigned short width  = camera.geometry().width();
    const unsigned short height = camera.geometry().height();

    // get the base of the input filename and the path
    const std::string output_base = std::regex_replace(in_file_path, std::regex("\\.[^.]*$"), "");

    // setup feedback to be provided on processing progression
    using namespace std::chrono_literals;
    auto log = MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline;
    log << "Writing to " << output_base << "{_cd,_trigger}.dat\n";
    const std::string message("Writing DAT file...");
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

    // write CD events to a DAT file
    const std::string cd_filename(output_base + "_cd.dat");
    Metavision::StreamLoggerAlgorithm stream(cd_filename, width, height);
    stream.enable(true, false);

    camera.cd().add_callback([&](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
        stream.process_events(ev_begin, ev_end, ev_begin->t);
        progress_feedback_fct();
    });

    // write external triggers events to a DAT file
    const std::string ext_trigger_filename(output_base + "_trigger.dat");
    Metavision::StreamLoggerAlgorithm stream_ext(ext_trigger_filename, width, height);
    stream_ext.enable(true, false);

    bool has_ext_trigger = false;
    try {
        camera.ext_trigger().add_callback(
            [&](const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
                stream_ext.process_events(begin, end, begin->t);
                has_ext_trigger = true;
            });
    } catch (Metavision::CameraException &) {}

    camera.start();

    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    };

    stream.close();
    stream_ext.close();

    log << "\rDone!                    " << std::endl;

    return 0;
}