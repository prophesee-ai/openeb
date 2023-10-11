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

// Example of using Metavision SDK Driver API to get information about a file.

#include <iostream>
#include <iomanip>
#include <sstream>
#include <array>
#include <thread>
#include <fstream>
#include <boost/format.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/camera_exception.h>

namespace po = boost::program_options;

std::string human_readable_rate(double rate) {
    std::ostringstream oss;
    if (rate < 1000) {
        oss << std::setprecision(0) << std::fixed << rate << " ev/s";
    } else if (rate < 1000 * 1000) {
        oss << std::setprecision(1) << std::fixed << (rate / 1000) << " Kev/s";
    } else if (rate < 1000 * 1000 * 1000) {
        oss << std::setprecision(1) << std::fixed << (rate / (1000 * 1000)) << " Mev/s";
    } else {
        oss << std::setprecision(1) << std::fixed << (rate / (1000 * 1000 * 1000)) << " Gev/s";
    }
    return oss.str();
}

std::string human_readable_time(Metavision::timestamp t) {
    std::ostringstream oss;
    std::array<std::string, 6> ls{"d", "h", "m", "s", "ms", "us"};
    std::array<int, 6> vs;
    vs[5] = (t % 1000);
    t /= 1000; // ms
    vs[4] = (t % 1000);
    t /= 1000; // s
    vs[3] = (t % 60);
    t /= 60; // m
    vs[2] = (t % 60);
    t /= 60; // h
    vs[1] = (t % 24);
    t /= 24; // d
    vs[0] = (t % 365);

    size_t i = 0;
    for (; i < 6 && vs[i] == 0; ++i) {}
    for (; i < 6; ++i) {
        oss << vs[i] << ls[i] << " ";
    }
    return oss.str();
}

int main(int argc, char *argv[]) {
    std::string in_file_path;

    const std::string program_desc("Application to get information about a file\n");

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

    std::ifstream ifs(in_file_path);
    if (!ifs.is_open()) {
        MV_LOG_ERROR() << "Unable to open file";
        return 1;
    }

    Metavision::Camera camera;
    try {
        auto hints = Metavision::FileConfigHints().real_time_playback(false);
        camera     = Metavision::Camera::from_file(in_file_path, hints);
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }

    struct EventType {
        enum { CD = 0, ExtTrigger, Count };
    };
    std::array<Metavision::timestamp, EventType::Count> first_ts, last_ts;
    std::array<size_t, EventType::Count> num_events;
    std::array<std::string, EventType::Count> label_events{"CD", "External triggers"};
    std::fill(first_ts.begin(), first_ts.end(), std::numeric_limits<Metavision::timestamp>::max());
    std::fill(last_ts.begin(), last_ts.end(), -1);
    std::fill(num_events.begin(), num_events.end(), 0);

    Metavision::timestamp duration = -1;
    try {
        auto &osc = camera.offline_streaming_control();
        while (!osc.is_ready()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        duration = osc.get_duration();
    } catch (Metavision::CameraException &e) {}

    try {
        Metavision::CD &cd = camera.cd();
        cd.add_callback(
            [&num_events, &first_ts, &last_ts](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
                num_events[EventType::CD] += std::distance(begin, end);
                first_ts[EventType::CD] = std::min(first_ts[EventType::CD], begin->t);
                last_ts[EventType::CD]  = std::max(last_ts[EventType::CD], (end - 1)->t);
            });
    } catch (...) {}

    try {
        Metavision::ExtTrigger &ext_trigger = camera.ext_trigger();
        ext_trigger.add_callback([&num_events, &first_ts, &last_ts](const Metavision::EventExtTrigger *begin,
                                                                    const Metavision::EventExtTrigger *end) {
            num_events[EventType::ExtTrigger] += std::distance(begin, end);
            first_ts[EventType::ExtTrigger] = std::min(first_ts[EventType::ExtTrigger], begin->t);
            last_ts[EventType::ExtTrigger]  = std::max(last_ts[EventType::ExtTrigger], (end - 1)->t);
        });
    } catch (...) {}

    camera.start();

    const std::string message("Analyzing file...");
    auto log = MV_LOG_INFO() << Metavision::Log::no_endline << Metavision::Log::no_space << message << std::flush;
    int dots = 0;

    // wait for the analysis to end
    while (camera.is_running()) {
        log << "\r" << message.substr(0, message.size() - 3 + dots) + std::string("   ").substr(0, 3 - dots)
            << std::flush;
        dots = (dots + 1) % 4;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    camera.stop();

    // if needed (i.e if OfflineStreamingControl facility is unavailable), update duration as the maximum timestamp ever
    // found
    for (size_t i = 0; i < EventType::Count; ++i) {
        duration = std::max(duration, last_ts[i]);
    }

    std::ostringstream tmp_oss;
    static constexpr size_t LineWidth = 100;
    tmp_oss << std::setfill('=') << std::setw(LineWidth) << "=";
    const std::string line_sep = tmp_oss.str();

    log << "\r" << line_sep << "\n\n";

    const std::string global_format("%-20s%s");
    log << boost::format(global_format) % "Name" % boost::filesystem::path(in_file_path).filename().string() << "\n";
    log << boost::format(global_format) % "Path" %
               boost::filesystem::canonical(boost::filesystem::path(in_file_path)).make_preferred().string()
        << "\n";
    log << boost::format(global_format) % "Duration" % human_readable_time(duration) << "\n";

    auto &config = camera.get_camera_configuration();
    if (!config.integrator.empty()) {
        log << boost::format(global_format) % "Integrator" % config.integrator << "\n";
    }

    if (!config.plugin_name.empty()) {
        log << boost::format(global_format) % "Plugin name" % config.plugin_name << "\n";
    }

    if (!config.data_encoding_format.empty()) {
        log << boost::format(global_format) % "Data encoding" % config.data_encoding_format << "\n";
    }

    auto &gen = camera.generation();
    tmp_oss   = std::ostringstream();
    tmp_oss << gen.version_major() << "." << gen.version_minor();
    log << boost::format(global_format) % "Camera generation" % tmp_oss.str() << "\n";

    if (!config.system_ID.empty()) {
        log << boost::format(global_format) % "Camera systemID" % config.system_ID << "\n";
    }

    if (!config.serial_number.empty()) {
        log << boost::format(global_format) % "Camera serial" % config.serial_number << "\n";
    }

    log << "\n" << line_sep << "\n\n";

    static constexpr size_t EventsLineWidth = 100;
    const std::string events_format("%-20s%-20s%-20s%-20s%-20s");
    tmp_oss = std::ostringstream();
    tmp_oss << std::setfill('-') << std::setw(EventsLineWidth) << "-";
    const std::string events_line_sep = tmp_oss.str();
    log << boost::format(events_format) % "Type of event" % "Number of events" % "First timestamp" % "Last timestamp" %
               "Average event rate"
        << "\n";
    log << events_line_sep << "\n";

    for (size_t i = 0; i < EventType::Count; ++i) {
        if (num_events[i] != 0) {
            log << boost::format(events_format) % label_events[i] % num_events[i] % first_ts[i] % last_ts[i] %
                       human_readable_rate(num_events[i] / (duration / 1.e6))
                << "\n";
        }
    }
    log << std::flush;

    return 0;
}
