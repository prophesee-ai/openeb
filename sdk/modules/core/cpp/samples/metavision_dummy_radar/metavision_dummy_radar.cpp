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

// Example of using Metavision SDK API for building a passive radar for moving
// objects
//
// It simply counts all events falling within vertical bins (groups of sensor columns) of the sensor (the number of
// bins being chosen by the user). Then, the event rate of each bin is computed from the event count and the selected
// period. The maximum event rate is found and if it falls within the operational event rate boundaries, it is assumed
// to correspond to the object of interest.
// To estimate a relative "distance", it is assumed that the closer the object, the more events will be generated
// (filling more the field of view). Thus, the "distance" to the camera is inversely proportional to the event rate
// of the bin with more events.
// Finally, the histogram is transformed into a "radar"-like shape to get better insight on the location of the target.

#include <boost/program_options.hpp>
#include <opencv2/highgui.hpp>

// Basic utils for camera streaming
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/utils/window.h>

// More advanced classes for event processing
#include <metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>

#include "activity_monitoring.h"
#include "radar_viewer.h"

namespace po = boost::program_options;

struct Params {
    std::string serial;
    std::string cam_config_path;
    std::string cam_calibration_path;
    std::string event_file_path;
    uint32_t delta_ts;
    uint32_t stc_threshold;
    uint16_t nbins;
    float min_ev_rate;
    float max_ev_rate;
    float camera_fov;
};

int main(int argc, char *argv[]) {
    /// [PROGRAM INPUTS]
    Params parameters;

    const std::string short_program_desc(
        "Example of using Metavision SDK API for processing events and visualizing some"
        "spatial representation of events.\n");
    po::options_description options_desc("Options");

    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i",    po::value<std::string>(&parameters.event_file_path), "Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
        ("input-camera-config,j", po::value<std::string>(&parameters.cam_config_path), "Path to a JSON file containing camera config settings to restore a camera state. Only works for live cameras.")
        ("accumulation-time,a",   po::value<uint32_t>(&parameters.delta_ts)->default_value(50000), "Accumulation time for which to display the radar plot.")
        ("min-ev-rate",   po::value<float>(&parameters.min_ev_rate)->default_value(1e5f), "Minimum event rate per bin.")
        ("max-ev-rate",   po::value<float>(&parameters.max_ev_rate)->default_value(3e6f), "Maximum event rate per bin.")
        ("cam-fov-deg",   po::value<float>(&parameters.camera_fov)->default_value(90.f), "Camera lateral FOV (in degrees).")
        ("nbins",   po::value<uint16_t>(&parameters.nbins)->default_value(8), "Number of bins describing the FOV.")
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

    if (parameters.nbins == 0) {
        MV_LOG_ERROR() << "The number of bins should be strictly positive.";
        return 2;
    }

    /// [CAMERA INITIALIZATION]
    // If the filename is set, then read from the file
    Metavision::Camera camera;
    if (parameters.event_file_path.empty()) {
        camera = Metavision::Camera::from_first_available();
        if (!parameters.cam_config_path.empty())
            camera.load(parameters.cam_config_path);
    } else {
        const auto cam_config = Metavision::FileConfigHints().real_time_playback(true);
        camera                = Metavision::Camera::from_file(parameters.event_file_path, cam_config);
    }

    /// [ALGO INITIALIZATION]
    // Get camera resolution
    const int camera_width  = camera.geometry().width();
    const int camera_height = camera.geometry().height();

    ActivityMonitor::Config config;
    config.n_bins            = parameters.nbins;
    config.accumulation_time = parameters.delta_ts;
    ActivityMonitor activity_monitor(config, camera_width);

    RadarViewer::Config conf;
    conf.n_bins_x    = parameters.nbins;
    conf.min_ev_rate = parameters.min_ev_rate;
    conf.max_ev_rate = parameters.max_ev_rate;
    conf.lateral_fov = parameters.camera_fov * M_PI / 180.f;
    RadarViewer radar_plot(conf, camera_width, camera_height);

    /// [DISPLAY WINDOW INITIALIZATION]

    // To render the frames, we create a window using the Window class of the UI module
    Metavision::Window window("Radar viewer", 2 * camera_width, camera_height, Metavision::BaseWindow::RenderMode::BGR);

    cv::Mat disp_frame(cv::Size(camera_width * 2, camera_height), CV_8UC3);

    auto frame_gen = Metavision::OnDemandFrameGenerationAlgorithm(camera_width, camera_height, parameters.delta_ts);

    // We set a callback on the windows to close it when the Escape or Q key is
    // pressed
    window.set_keyboard_callback(
        [&window, &disp_frame](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
            if (action == Metavision::UIAction::RELEASE &&
                (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                window.set_close_flag();
            }
            if (action == Metavision::UIAction::RELEASE && (key == Metavision::UIKeyEvent::KEY_R)) {
                cv::imwrite("radar.png", disp_frame);
            }
        });

    /// [PROCESSING DESCRIPTION]
    cv::Rect evts_roi(0, 0, camera_width, camera_height);
    cv::Rect radar_roi(camera_width, 0, camera_width, camera_height);

    cv::Mat radar_frame = disp_frame(radar_roi);
    cv::Mat evts_frame  = disp_frame(evts_roi);

    // Initialize the slicer which is gonna slice the events into same duration
    // buffers of events
    std::vector<float> ev_rate_per_bin;
    using Slicer = Metavision::EventBufferReslicerAlgorithm;
    Slicer slicer(
        [&](Slicer::ConditionStatus, Metavision::timestamp ts, std::size_t) {
            activity_monitor.get_ev_rate_per_bin(ev_rate_per_bin);
            radar_plot.compute_view(ev_rate_per_bin, radar_frame);
            frame_gen.generate(ts, evts_frame, false);
            window.show(disp_frame);
            activity_monitor.reset();
        },
        Slicer::Condition::make_n_us(parameters.delta_ts));

    // Update the frame to be displayed as well as the event histogram used for the radar plot using a callback on CD
    // events
    camera.cd().add_callback([&](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
        slicer.process_events(ev_begin, ev_end, [&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            if (begin == end)
                return;

            frame_gen.process_events(begin, end);
            activity_monitor.process_events(begin, end);
        });
    });

    // Start the camera
    camera.start();

    // Keep running until the camera is off, the recording is finished or the
    // escape key was pressed
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
