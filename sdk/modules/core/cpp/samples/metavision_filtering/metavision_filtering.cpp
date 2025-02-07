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

// This code sample demonstrates how to use Metavision Core SDK utility to display and filter events.
// It shows how to capture the keys pressed in a display window so as to modify the behavior of the filters while the
// camera is running.

#include <atomic>
#include <functional>
#include <vector>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/algorithms/polarity_filter_algorithm.h>
#include <metavision/sdk/core/algorithms/roi_filter_algorithm.h>
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/stream/camera_exception.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string event_file_path;

    const std::string short_program_desc("Code sample showing how to create a simple application to filter"
                                         " and display events.\n");

    const std::string long_program_desc(short_program_desc + "Available keyboard options:\n"
                                                             "  - r - toggle the ROI filter algorithm\n"
                                                             "  - p - show only events of positive polarity\n"
                                                             "  - n - show only events of negative polarity\n"
                                                             "  - a - show all events\n"
                                                             "  - q - quit the application\n");

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
        MV_LOG_ERROR() << short_program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << short_program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    MV_LOG_INFO() << long_program_desc;

    // Construct a camera from a recording or a live stream
    Metavision::Camera cam;
    std::atomic<bool> should_stop = false;
    if (!event_file_path.empty()) {
        cam = Metavision::Camera::from_file(event_file_path);
    } else {
        cam = Metavision::Camera::from_first_available();
    }
    const unsigned short width  = cam.geometry().get_width();
    const unsigned short height = cam.geometry().get_height();

    Metavision::RoiFilterAlgorithm roi_filter(150, 150, width - 150, height - 150, false);
    std::atomic<bool> roi_filter_enabled = false;
    Metavision::PolarityFilterAlgorithm pol_filter(0);
    std::atomic<bool> pol_filter_enabled = false;

    // Generating a frame from filtered events using accumulation time of 30ms
    Metavision::PeriodicFrameGenerationAlgorithm frame_generator(width, height, 30000);

    try {
        cam.cd().add_callback(
            [&roi_filter_enabled, &roi_filter,
             &pol_filter_enabled, &pol_filter,
             &frame_generator](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
                std::vector<Metavision::EventCD> roi_filter_out;
                if (roi_filter_enabled) {
                    roi_filter.process_events(begin, end, std::back_inserter(roi_filter_out));
                    begin = roi_filter_out.data();
                    end = begin + roi_filter_out.size();
                }

                std::vector<Metavision::EventCD> pol_filter_out;
                if (pol_filter_enabled) {
                    pol_filter.process_events(begin, end, std::back_inserter(pol_filter_out));
                    begin = pol_filter_out.data();
                    end = begin + pol_filter_out.size();
                }

                frame_generator.process_events(begin, end);
            });
    } catch (const Metavision::CameraException &e) {
        MV_LOG_ERROR() << "Unexpected error: " << e.what();
    }

    Metavision::Window window("CD events", width, height, Metavision::Window::RenderMode::BGR);

    window.set_keyboard_callback(
        [&roi_filter_enabled, &pol_filter_enabled, &pol_filter, &cam, &should_stop](Metavision::UIKeyEvent key,
                                                                                    int scancode,
                                                                                    Metavision::UIAction action, int mods) {
        if (action == Metavision::UIAction::RELEASE) {
            switch (key) {
            case Metavision::UIKeyEvent::KEY_A:
                // show all events
                pol_filter_enabled = false;
                break;
            case Metavision::UIKeyEvent::KEY_N:
                // show only negative events
                pol_filter_enabled = true;
                pol_filter.set_polarity(0);
                break;
            case Metavision::UIKeyEvent::KEY_P:
                // show only positive events
                pol_filter_enabled = true;
                pol_filter.set_polarity(1);
                break;
            case Metavision::UIKeyEvent::KEY_R:
                // toggle ROI filter
                roi_filter_enabled = !roi_filter_enabled;
                break;
            case Metavision::UIKeyEvent::KEY_ESCAPE:
            case Metavision::UIKeyEvent::KEY_Q:
                should_stop = true;
                break;
            default:
                break;
            }
        }
    });

    frame_generator.set_output_callback(
        [&window](Metavision::timestamp t, cv::Mat &frame_data) {
            if (!frame_data.empty())
                window.show(frame_data);
        });

    cam.start();
    while (!should_stop && cam.is_running()) {
        Metavision::EventLoop::poll_and_dispatch();
    }
    cam.stop();

    return 0;
}
