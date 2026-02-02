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

// This code sample demonstrates how to use Metavision SDK Stream to convert an event file
// to separate CSV files for events and triggers

#include <fstream>
#include <thread>
#include <regex>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/stream/camera_exception.h>
#include <metavision/sdk/stream/camera.h>

class CSVWriter {
public:
    CSVWriter(const std::string &filename) : filename_(filename) {}
    // these functions will be associated to the camera callback
    void write_cd(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        ensure_file_opened();
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            ofs_ << ev->x << "," << ev->y << "," << ev->p << "," << ev->t << "\n";
        }
    }

    void write_trigger(const Metavision::EventExtTrigger *begin, const Metavision::EventExtTrigger *end) {
        ensure_file_opened();
        for (const Metavision::EventExtTrigger *ev = begin; ev != end; ++ev) {
            ofs_ << ev->p << "," << ev->id << "," << ev->t << "\n";
        }
    }

    bool is_open() const {
        return ofs_.is_open();
    }

private:
    void ensure_file_opened() {
        if (!ofs_.is_open()) {
            ofs_.open(filename_);
            if (!ofs_.is_open()) {
                MV_LOG_ERROR() << "Unable to write in" << filename_;
            }
        }
    }

    std::ofstream ofs_;
    std::string filename_;
};

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string event_file_path;
    std::string cd_file_path;
    std::string trigger_file_path;
    bool disable_ts_shifting = false;

    const std::string program_desc(
        "Code sample demonstrating how to use Metavision SDK Stream to convert an event file "
        "to CSV files for events and triggers.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i", po::value<std::string>(&event_file_path)->required(), "Path to the input event file.")
        ("output-file,o", po::value<std::string>(&cd_file_path)->default_value(""),
         "Path to the output CSV file for CD events. If trigger events are found, this path will be used as the base name for the output trigger CSV file.")
        ("disable-timestamp-shifting,d", po::bool_switch(&disable_ts_shifting), "Disable shifting all event timestamps in a RAW file to be relative to the timestamp of the first event, preserving their original absolute values")
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

    // build the output files paths
    if (cd_file_path.empty()) {
        cd_file_path = std::filesystem::path(event_file_path).replace_extension(".csv").string();
    }
    trigger_file_path = std::filesystem::path(cd_file_path).replace_extension().string() + "_triggers.csv";

    // open the input file
    Metavision::Camera cam;
    try {
        auto hints = Metavision::FileConfigHints().real_time_playback(false);
        hints.set("time_shift", !disable_ts_shifting);
        cam = Metavision::Camera::from_file(event_file_path, hints);
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }

    // setup feedback to be provided on processing progression
    using namespace std::chrono_literals;
    auto log = MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline;
    log << "Writing to " << cd_file_path << "\n";
    const std::string message("Writing CSV file...");
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

    // setup writers
    CSVWriter cd_writer(cd_file_path);
    CSVWriter trigger_writer(trigger_file_path);

    // to write the CD events, we add a callback that will be called periodically to give access to the latest events
    cam.cd().add_callback([&](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
        progress_feedback_fct();
        cd_writer.write_cd(ev_begin, ev_end);
    });

    // callback for External Trigger events
    cam.ext_trigger().add_callback(
        [&](const Metavision::EventExtTrigger *ev_begin, const Metavision::EventExtTrigger *ev_end) {
            trigger_writer.write_trigger(ev_begin, ev_end);
    });

    cam.start();

    // keep running until the recording is finished
    while (cam.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };

    log << "\rDone!                    " << std::endl;
    if (trigger_writer.is_open()) {
        MV_LOG_INFO() << "Trigger events written to" << trigger_file_path;
    };

    return 0;
}
