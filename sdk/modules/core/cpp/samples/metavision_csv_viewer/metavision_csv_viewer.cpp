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

// This code sample demonstrates how to use Metavision Core SDK to display events read from a CSV
// formatted event-based file, such as one produced by the sample metavision_file_to_csv.

#include <atomic>
#include <iostream>
#include <functional>
#include <fstream>
#include <optional>
#include <thread>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/object_pool.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

// A stage producing events from a CSV file with events written in the following format:
// x1,y1,t1,p1
// x2,y2,t2,p2
// ...
// xn,yn,tn,pn

class CSVReader {
public:
    using EventBuffer = std::vector<Metavision::EventCD>;
    using EventBufferPool = Metavision::SharedObjectPool<EventBuffer>;
    using EventBufferPtr = EventBufferPool::ptr_type;
    using OutputCallback = std::function<void(const Metavision::EventCD *, const Metavision::EventCD *)>;

    CSVReader(const std::string &filename) : ifs_(filename) {
        if (!ifs_.is_open()) {
            MV_LOG_ERROR() << "Unable to open " << filename;
            throw std::runtime_error("Unable to open " + filename);
        }
        if (!parse_csv_header()) {
            MV_LOG_ERROR() << "Error while parsing header of " << filename;
            throw std::runtime_error("Error while parsing header of " + filename);
        }

        cur_cd_buffer_ = cd_buffer_pool_.acquire();
        cur_cd_buffer_->clear();
    }

    void read() {
        std::string line;
        while (!done_ && std::getline(ifs_, line)) {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (std::getline(iss, token, ',')) {
                tokens.push_back(token);
            }
            if (tokens.size() != 4) {
                MV_LOG_ERROR() << "Invalid line : <" << line << "> ignored";
            } else {
                if (cur_cd_buffer_->size() > 5000) {
                    output_cb_(cur_cd_buffer_->data(), cur_cd_buffer_->data() + cur_cd_buffer_->size());
                    cur_cd_buffer_ = cd_buffer_pool_.acquire();
                    cur_cd_buffer_->clear();
                }
                cur_cd_buffer_->emplace_back(static_cast<unsigned short>(std::stoul(tokens[0])),
                                             static_cast<unsigned short>(std::stoul(tokens[1])),
                                             static_cast<short>(std::stoi(tokens[2])), std::stoll(tokens[3]));
            }
        }
    }

    void stop() {
        done_ = true;
    }

    void set_output_callback(const OutputCallback &out_cb) {
        output_cb_ = out_cb;
    }

    std::optional<int> get_width() const {
        return width_;
    }
    std::optional<int> get_height() const {
        return height_;
    }

private:
    bool read_cd_csv_header_line() {
        std::string line;
        if (std::getline(ifs_, line)) {
            std::istringstream iss(line);
            std::string key, value;
            std::vector<std::string> values;
            iss.ignore(1); // ignore leading '%'
            std::getline(iss, key, ':');
            while (std::getline(iss, value, ',')) {
                values.push_back(value);
            }
            if (key == "geometry") {
                if (values.size() == 2) {
                    width_  = std::stoi(values[0]);
                    height_ = std::stoi(values[1]);
                } else {
                    MV_LOG_ERROR() << "Ignoring invalid header line for key geometry, expected "
                                      "\"%geometry:<width>,<height>\", got: \""
                                   << line << "\"";
                }
            }
        }
        return ifs_.good();
    }

    bool parse_csv_header() {
        while (ifs_.peek() == '%') {
            if (!read_cd_csv_header_line()) {
                return false;
            }
        }
        return true;
    }

    std::optional<int> width_, height_;
    std::atomic<bool> done_ = false;
    std::thread reading_thread_;
    std::ifstream ifs_;
    EventBufferPool cd_buffer_pool_;
    EventBufferPtr cur_cd_buffer_;
    OutputCallback output_cb_;
};

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string in_csv_file_path;
    int width, height;

    const std::string program_desc(
        "Code sample demonstrating how to use Metavision SDK to display events from a CSV file.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-csv-file,i", po::value<std::string>(&in_csv_file_path)->required(), "Path to input CSV file")
        ("width",            po::value<int>(&width)->default_value(1280), "Width of the sensor associated to the CSV file")
        ("height",           po::value<int>(&height)->default_value(720), "Height of the sensor associated to the CSV file")
        ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    /// Pipeline
    //
    //  0 (Csv Reading) -->-- 1 (Frame Generation) -->-- 2 (Display)
    //

    std::atomic<bool> should_stop = false;
    CSVReader csv_reader(in_csv_file_path);
    if (auto width_opt = csv_reader.get_width()) {
        width = *width_opt;
    }
    if (auto height_opt = csv_reader.get_height()) {
        height = *height_opt;
    }

    Metavision::PeriodicFrameGenerationAlgorithm frame_generator(width, height, 10000);
    csv_reader.set_output_callback([&frame_generator](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
                                       frame_generator.process_events(begin, end);
                                   });

    Metavision::Window window("CD events", width, height, Metavision::Window::RenderMode::BGR);
    window.set_keyboard_callback([&should_stop, &csv_reader](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                                     if (action == Metavision::UIAction::RELEASE) {
                                         switch (key) {
                                         case Metavision::UIKeyEvent::KEY_ESCAPE:
                                         case Metavision::UIKeyEvent::KEY_Q:
                                             should_stop = true;
                                             csv_reader.stop();
                                             break;
                                         }
                                     }
                                 });

    frame_generator.set_output_callback(
        [&window](Metavision::timestamp t, cv::Mat &frame_data) {
            if (!frame_data.empty())
                window.show(frame_data);
        });

    auto t = std::thread([&csv_reader]() { csv_reader.read(); });
    while (!t.joinable()) {
    }

    while (!should_stop) {
        Metavision::EventLoop::poll_and_dispatch();
    }

    t.join();

    return 0;
}
