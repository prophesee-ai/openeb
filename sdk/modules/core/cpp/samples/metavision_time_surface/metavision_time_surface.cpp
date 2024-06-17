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

// Example of using Metavision SDK API for visualizing Time Surface of events

#include <boost/program_options.hpp>

// Basic utils for camera streaming
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>

// More advanced classes for event processing
#include <metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h>
#include <metavision/sdk/core/utils/mostrecent_timestamp_buffer.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string serial;
    std::string cam_config_path;
    std::string event_file_path;
    uint32_t delta_ts;

    const std::string short_program_desc(
        "Example of using Metavision SDK Core API for visualizing Time Surface of events.\n");
    po::options_description options_desc("Options");

    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i",    po::value<std::string>(&event_file_path), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("input-camera-config,j", po::value<std::string>(&cam_config_path), "Path to a JSON file containing camera config settings to restore a camera state. Only works for live cameras.")
        ("accumulation-time,a",   po::value<uint32_t>(&delta_ts)->default_value(50000), "Accumulation time for which to display the Time Surface.")
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

    // If the filename is set, then read from the file
    Metavision::Camera camera;
    try {
        if (event_file_path.empty()) {
            camera = Metavision::Camera::from_first_available();
            if (!cam_config_path.empty())
                camera.load(cam_config_path);
        } else {
            camera = Metavision::Camera::from_file(event_file_path, Metavision::FileConfigHints().real_time_playback(true));
        }
    } catch (const Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 2;
    }

    // Get camera resolution
    const int camera_width  = camera.geometry().width();
    const int camera_height = camera.geometry().height();

    // To render the frames, we create a window using the Window class of the UI
    // module
    Metavision::Window window("Metavision Time Surface", camera_width, camera_height,
                              Metavision::BaseWindow::RenderMode::BGR);

    // We set a callback on the windows to close it when the Escape or Q key is
    // pressed
    window.set_keyboard_callback(
        [&window](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
            if (action == Metavision::UIAction::RELEASE &&
                (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                window.set_close_flag();
            }
        });

    // Create a MostRecentTimestampBuffer to store the last timestamp of each
    // pixel and initialize all elements to zero
    Metavision::MostRecentTimestampBuffer time_surface(camera_height, camera_width, 1);
    time_surface.set_to(0);
    // Create a variable to store the latest timestamp
    Metavision::timestamp last_time = 0;
    // Create cv::Mat to store the time surface and the heatmap

    // Initialize the slicer which is gonna slice the events into same duration
    // buffers of events
    cv::Mat heatmap, time_surface_gray;
    using Slicer = Metavision::EventBufferReslicerAlgorithm;
    Slicer slicer(
        [&](Slicer::ConditionStatus, Metavision::timestamp, std::size_t) {
            // Generate the time surface from MostRecentTimestampBuffer
            time_surface.generate_img_time_surface(last_time, delta_ts, time_surface_gray);
            // Apply a colormap to the time surface and display the new frame
            cv::applyColorMap(time_surface_gray, heatmap, cv::COLORMAP_JET);
            window.show(heatmap);
        },
        Slicer::Condition::make_n_us(delta_ts));

    // Update the time surface using a callback on CD events
    camera.cd().add_callback([&](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
        slicer.process_events(ev_begin, ev_end, [&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            if (begin == end)
                return;

            last_time = std::prev(end)->t;
            for (auto it = begin; it != end; ++it)
                time_surface.at(it->y, it->x) = it->t;
        });
    });

    // Start the camera
    camera.start();

    // Keep running until the recording is finished, the escape or 'q' key was pressed, or the window was closed
    while (camera.is_running() && !window.should_close()) {
        // We poll events (keyboard, mouse etc.) from the system with a 20ms sleep
        // to avoid using 100% of a CPU's core and we push them into the window
        // where the callback on the escape key will ask the windows to close
        static constexpr std::int64_t kSleepPeriodMs = 20;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }

    camera.stop();

    return 0;
}
