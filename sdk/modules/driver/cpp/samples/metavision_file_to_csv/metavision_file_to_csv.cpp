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

// This code sample demonstrates how to use Metavision SDK Driver to convert an event-based
// file to a CSV formatted event-based file.

#include <fstream>
#include <thread>
#include <regex>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/driver/camera_exception.h>
#include <metavision/sdk/driver/camera.h>

class CSVWriter {
public:
    CSVWriter(std::string &filename) : ofs_(filename) {
        if (!ofs_.is_open()) {
            MV_LOG_ERROR() << "Unable to write in" << filename;
        }
    };
    // this function will be associated to the camera callback
    void write(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        for (const Metavision::EventCD *ev = begin; ev != end; ++ev) {
            ofs_ << ev->x << "," << ev->y << "," << ev->p << "," << ev->t << "\n";
        }
    }

private:
    std::ofstream ofs_;
};

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string in_file_path;
    std::string out_file_path;

    const std::string program_desc(
        "Code sample demonstrating how to use Metavision SDK Driver to convert a file to a CSV formatted"
        " file.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i", po::value<std::string>(&in_file_path)->required(), "Path to the input file.")
        ("output-file,o", po::value<std::string>(&out_file_path)->default_value(""), "Path to the output file.")
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

    // get the base of the input filename and the path
    if (out_file_path.empty()) {
        const std::string output_base = std::regex_replace(in_file_path, std::regex("\\.[^.]*$"), "");
        out_file_path                 = output_base + ".csv";
    }

    // open the file that was passed
    Metavision::Camera cam;
    try {
        cam = Metavision::Camera::from_file(in_file_path, Metavision::FileConfigHints().real_time_playback(false));
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }

    // setup feedback to be provided on processing progression
    using namespace std::chrono_literals;
    auto log = MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline;
    log << "Writing to " << out_file_path << "\n";
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

    // to write the events, we add a callback that will be called periodically to give access to the latest events
    CSVWriter writer(out_file_path);
    cam.cd().add_callback([&](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
        writer.write(ev_begin, ev_end);
        progress_feedback_fct();
    });

    cam.start();

    // keep running until the recording is finished
    while (cam.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };
    log << "\rDone!                    " << std::endl;

    return 0;
}
