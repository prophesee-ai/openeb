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

// This code sample demonstrates how to use Metavision Core SDK to filter events and show a frame
// combining unfiltered and filtered events.

#include <atomic>
#include <chrono>
#include <functional>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/algorithms/polarity_filter_algorithm.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/frame_composer.h>
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string event_file_path;

    const std::string program_desc("Code sample demonstrating how to use Metavision SDK CV to filter events\n"
                                   "and show a frame combining unfiltered and filtered events.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i", po::value<std::string>(&event_file_path), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
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

    // Construct a camera from a file or a live stream
    Metavision::Camera cam;
    std::atomic<bool> should_stop = false;
    if (!event_file_path.empty()) {
        cam = Metavision::Camera::from_file(event_file_path);
    } else {
        cam = Metavision::Camera::from_first_available();
    }
    const unsigned short width  = cam.geometry().get_width();
    const unsigned short height = cam.geometry().get_height();

    const Metavision::timestamp event_buffer_duration_ms = 2;
    const uint32_t accumulation_time_ms                  = 10;
    const int display_fps                                = 100;

    //
    //  0 (Camera) ---------------->---------------- 1 (Polarity Filter)
    //  |                                            |
    //  v                                            v
    //  |                                            |
    //  2 (Frame Generation)                         3 (Frame Generation)
    //  |                                            |
    //  v                                            v
    //  |------>-----  4 (Frame Composer)  ----<-----|
    //                 |
    //                 v
    //                 |
    //                 5 (Display)
    //

    Metavision::FrameComposer frame_composer;

    // Left frame, plain output from the sensor
    const auto left_image_ref = frame_composer.add_new_subimage_parameters(0, 0, {width, height}, Metavision::FrameComposer::GrayToColorOptions());
    Metavision::PeriodicFrameGenerationAlgorithm left_frame_generator(width, height, accumulation_time_ms * 1000);
    cam.cd().add_callback([&left_frame_generator](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
                              left_frame_generator.process_events(begin, end);
                          });
    left_frame_generator.set_output_callback(
        [&left_image_ref, &frame_composer](Metavision::timestamp t, cv::Mat &frame_data) {
            if (!frame_data.empty()) {
                frame_composer.update_subimage(left_image_ref, frame_data);
            }
        });

    // Right frame, output only positive events
    const auto right_image_ref = frame_composer.add_new_subimage_parameters(width + 10, 0, {width, height}, Metavision::FrameComposer::GrayToColorOptions());
    Metavision::PeriodicFrameGenerationAlgorithm right_frame_generator(width, height, accumulation_time_ms * 1000);
    Metavision::PolarityFilterAlgorithm pol_filter(1);
    cam.cd().add_callback([&pol_filter, &right_frame_generator](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
                              std::vector<Metavision::EventCD> pol_filter_out;
                              pol_filter.process_events(begin, end, std::back_inserter(pol_filter_out));
                              right_frame_generator.process_events(pol_filter_out.begin(), pol_filter_out.end());
                          });
    right_frame_generator.set_output_callback(
        [&right_image_ref, &frame_composer](Metavision::timestamp t, cv::Mat &frame_data) {
            if (!frame_data.empty()) {
                frame_composer.update_subimage(right_image_ref, frame_data);
            }
        });


    Metavision::Window window("CD & noise filtered CD events", frame_composer.get_total_width(), frame_composer.get_total_height(),
                              Metavision::Window::RenderMode::BGR);
    window.set_keyboard_callback([&should_stop](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
        if (action == Metavision::UIAction::RELEASE) {
            switch (key) {
            case Metavision::UIKeyEvent::KEY_ESCAPE:
            case Metavision::UIKeyEvent::KEY_Q:
                should_stop = true;
                break;
            }
        }
    });

    const auto period_us = std::chrono::microseconds(1000000 / display_fps);
    auto last_frame_update = std::chrono::high_resolution_clock::now() - period_us;
    cam.start();
    while (!should_stop && cam.is_running()) {
        if (std::chrono::high_resolution_clock::now() - last_frame_update >= period_us
            && !frame_composer.get_full_image().empty()) {
            window.show(frame_composer.get_full_image());
            last_frame_update = std::chrono::high_resolution_clock::now();
        }
        Metavision::EventLoop::poll_and_dispatch();
    }
    cam.stop();

    return 0;
}
