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

#include <atomic>
#include <regex>
#include <thread>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/raw_event_file_writer.h>
#include <metavision/sdk/driver/hdf5_event_file_writer.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string in_file_path;
    std::string out_file_path;
    double start, end;

    const std::string program_desc("Cuts a file between <start> and <end> seconds where <start> and <end> are "
                                   "offsets from the beginning of the file, expressed as floating point numbers.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i",    po::value<std::string>(&in_file_path)->required(), "Path to input file.")
        ("output-file,o",   po::value<std::string>(&out_file_path)->required(), "Path to output file.")
        ("start,s",   po::value<double>(&start)->required(), "The start of the required sequence in seconds.")
        ("end,e",     po::value<double>(&end)->required(), "The end of the required sequence in seconds.")
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

    // convert start and end to microseconds
    Metavision::timestamp start_ts = static_cast<Metavision::timestamp>(start * 1000000);
    Metavision::timestamp end_ts   = static_cast<Metavision::timestamp>(end * 1000000);
    if (end <= start) {
        MV_LOG_ERROR() << "End time" << end << "is less than or equal to start" << start;
        return 1;
    }

    Metavision::Camera camera;
    try {
        Metavision::FileConfigHints hints = Metavision::FileConfigHints().real_time_playback(false);
        camera                            = Metavision::Camera::from_file(in_file_path, hints);
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }

    std::unique_ptr<Metavision::EventFileWriter> writer;
    std::string out_file_ext;
    std::smatch ext_match;
    if (std::regex_search(in_file_path, ext_match, std::regex("\\.[^.]*$"))) {
        out_file_ext = ext_match.str();
    }
    if (out_file_ext == ".raw") {
        try {
            camera.raw_data();
            writer = std::make_unique<Metavision::RAWEventFileWriter>(out_file_path);
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << "Unable to cut to RAW from a file that does not contain RAW data";
            return 1;
        }
    } else if (out_file_ext == ".hdf5") {
        writer = std::make_unique<Metavision::HDF5EventFileWriter>(out_file_path);
    } else {
        MV_LOG_ERROR() << "Unsupported file extension for output :" << out_file_ext;
        return 1;
    }

    writer->add_metadata_map_from_camera(camera);
    writer->add_metadata("time_shift", std::to_string(start_ts));

    std::atomic<bool> done(false);
    if (out_file_ext == ".raw") {
        // Handle cutting to a RAW file
        camera.raw_data().add_callback(
            [&done, &camera, &writer, start_ts, end_ts](const std::uint8_t *ptr, size_t size) {
                if (done) {
                    return;
                }
                auto *raw_writer = static_cast<Metavision::RAWEventFileWriter *>(writer.get());
                auto last_ts     = camera.get_last_timestamp();
                if (last_ts >= start_ts && last_ts <= end_ts) {
                    raw_writer->add_raw_data(ptr, size);
                }
                if (last_ts > end_ts) {
                    done = true;
                }
            });
    } else {
        camera.cd().add_callback([&done, &writer, start_ts, end_ts](const Metavision::EventCD *begin,
                                                                    const Metavision::EventCD *end) {
            if (done) {
                return;
            }
            auto *ev_begin =
                std::lower_bound(begin, end, start_ts,
                                 [](const Metavision::EventCD &ev, Metavision::timestamp ts) { return ev.t < ts; });
            auto *ev_end = std::lower_bound(
                begin, end, end_ts, [](const Metavision::EventCD &ev, Metavision::timestamp ts) { return ev.t < ts; });
            if (ev_begin != ev_end) {
                writer->add_events(ev_begin, ev_end);
            }
            if (std::prev(end)->t >= end_ts) {
                done = true;
            }
        });
        try {
            camera.ext_trigger().add_callback(
                [&done, &writer, start_ts, end_ts](const Metavision::EventExtTrigger *begin,
                                                   const Metavision::EventExtTrigger *end) {
                    if (done) {
                        return;
                    }
                    auto *ev_begin = std::lower_bound(
                        begin, end, start_ts,
                        [](const Metavision::EventExtTrigger &ev, Metavision::timestamp ts) { return ev.t < ts; });
                    auto *ev_end = std::lower_bound(
                        begin, end, end_ts,
                        [](const Metavision::EventExtTrigger &ev, Metavision::timestamp ts) { return ev.t < ts; });
                    if (ev_begin != ev_end) {
                        writer->add_events(ev_begin, ev_end);
                    }
                });
        } catch (Metavision::CameraException &e) {}
    }

    if (out_file_ext != ".raw") {
        // Try to seek to start time if supported
        // When writing RAW data, we should not do this, we will miss RAW data that comes before the
        // first CD event
        try {
            while (!camera.offline_streaming_control().is_ready()) {
                std::this_thread::yield();
            }
            camera.offline_streaming_control().seek(start_ts);
        } catch (...) {}
    }

    camera.start();

    using namespace std::chrono_literals;
    auto log = MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline;
    const std::string message("Cutting file...");
    int dots       = 0;
    auto last_time = std::chrono::high_resolution_clock::now();

    while (camera.is_running() && !done) {
        const auto time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(time - last_time) > 500ms) {
            last_time = time;
            log << "\r" << message.substr(0, message.size() - 3 + dots) + std::string("   ").substr(0, 3 - dots)
                << std::flush;
            dots = (dots + 1) % 4;
        }
    }

    camera.stop();
    writer->close();

    MV_LOG_INFO() << "\rOutput saved in file" << out_file_path;

    return 0;
}
