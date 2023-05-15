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

// Example of using Metavision SDK Core API for visualizing Time Surface of events

#include <boost/program_options.hpp>
#include <mutex>
#include <metavision/sdk/core/utils/mostrecent_timestamp_buffer.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string serial;
    std::string biases_file;
    std::string in_file_path;

    const std::string short_program_desc(
        "Example of using Metavision SDK Core API for visualizing Time Surface of events.\n");
    po::options_description options_desc("Options");

    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("serial,s",          po::value<std::string>(&serial),"Serial ID of the camera. This flag is incompatible with flag '--input-file'.")
        ("input-file,i",      po::value<std::string>(&in_file_path), "Path to input file. If not specified, the camera live stream is used.")
        ("biases,b",          po::value<std::string>(&biases_file), "Path to a biases file. If not specified, the camera will be configured with the default biases.")
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

    Metavision::Camera camera;

    // if the filename is set, then read from the file
    if (!in_file_path.empty()) {
        if (!serial.empty()) {
            MV_LOG_ERROR() << "Options --serial and --input-file are not compatible.";
            return 1;
        }

        try {
            camera =
                Metavision::Camera::from_file(in_file_path, Metavision::FileConfigHints().real_time_playback(true));

        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 2;
        }
    // otherwise, set the input source to a camera
    } else {
        try {
            if (!serial.empty()) {
                camera = Metavision::Camera::from_serial(serial);
            } else {
                camera = Metavision::Camera::from_first_available();
            }

            if (biases_file != "") {
                camera.biases().set_from_file(biases_file);
            }
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 3;
        }
    }

    // get camera resolution
    int camera_width  = camera.geometry().width();
    int camera_height = camera.geometry().height();

    // create a MostRecentTimestampBuffer to store the last timestamp of each pixel and initialize all elements to zero
    Metavision::MostRecentTimestampBuffer time_surface(camera_height, camera_width, 1);
    time_surface.set_to(0);

    // we use a mutex to control concurrent accesses to the time surface
    std::mutex frame_mutex;
    // create a variable where to store the latest timestamp
    Metavision::timestamp last_time = 0;
    // update the time surface using a callback on CD events
    camera.cd().add_callback([&time_surface, &frame_mutex, &last_time](const Metavision::EventCD *ev_begin,
                                                                       const Metavision::EventCD *ev_end) {
        for (auto it = ev_begin; it != ev_end; ++it) {
            std::unique_lock<std::mutex> lock(frame_mutex);
            time_surface.at(it->y, it->x) = it->t;
            last_time                     = it->t;
        }
    });

    // to render the frames, we create a window using the Window class of the UI module
    Metavision::Window window("Metavision Time Surface", camera_width, camera_height,
                              Metavision::BaseWindow::RenderMode::BGR);

    // we set a callback on the windows to close it when the Escape or Q key is pressed
    window.set_keyboard_callback(
        [&window](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
            if (action == Metavision::UIAction::RELEASE &&
                (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                window.set_close_flag();
            }
        });

    // create cv::Mat to store the time surface and the heatmap
    cv::Mat heatmap, time_surface_gray;

    // start the camera
    camera.start();

    // keep running until the camera is off, the recording is finished or the escape key was pressed
    while (camera.is_running() && !window.should_close()) {
        if (!time_surface.empty()) {
            std::unique_lock<std::mutex> lock(frame_mutex);
            // generate the time surface from MostRecentTimestampBuffer
            time_surface.generate_img_time_surface(last_time, 10000, time_surface_gray);
            // apply a colormap to the time surface and display the new frame
            cv::applyColorMap(time_surface_gray, heatmap, cv::COLORMAP_JET);
            window.show(heatmap);
        }
        // we poll events (keyboard, mouse etc.) from the system with a 20ms sleep to avoid using 100% of a CPU's core
        // and we push them into the window where the callback on the escape key will ask the windows to close
        static constexpr std::int64_t kSleepPeriodMs = 20;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }

    camera.stop();
}
