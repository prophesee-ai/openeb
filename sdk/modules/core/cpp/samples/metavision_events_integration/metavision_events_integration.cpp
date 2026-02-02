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

// Example of using Metavision SDK Core API for integrating events in a simple way into a grayscale-like image

#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h>
#include <metavision/sdk/core/algorithms/events_integration_algorithm.h>
#include <metavision/sdk/core/algorithms/contrast_map_generation_algorithm.h>
#include <metavision/sdk/core/algorithms/on_demand_frame_generation_algorithm.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string serial;
    std::string cam_config_path;
    std::string in_file_path;
    Metavision::timestamp period_us;
    uint32_t decay_time;
    float contrast_on, contrast_off, diffusion_weight;
    int integration_blur_radius, tonemapping_max_ev_count;
    std::string output_video_path;

    const std::string short_program_desc(
        "Example of using Metavision SDK Core API for integrating events into a grayscale-like image.\n");
    po::options_description options_desc("Options");

    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("serial,s",          po::value<std::string>(&serial)->default_value(""),"Serial ID of the camera. If empty, will open the first camera found. This flag is incompatible with flag '--input-file'.")
        ("input-file,i",      po::value<std::string>(&in_file_path)->default_value(""), "Path to input file. If not specified, the camera live stream is used.")
        ("input-camera-config,j", po::value<std::string>(&cam_config_path)->default_value(""), "Path to a JSON file containing camera config settings to restore a camera state. Only works for live cameras.")
        ("period,p", po::value<Metavision::timestamp>(&period_us)->default_value(30'000), "Period for the generation of the integrated event frames, in us.")
        ("decay-time,d", po::value<uint32_t>(&decay_time)->default_value(100'000), "Decay time after which integrated frame tends back to neutral gray. This needs to be adapted to the scene dynamics.")
        ("blur-radius,r", po::value<int>(&integration_blur_radius)->default_value(1), "Gaussian blur radius to be used to smooth integrated intensities.")
        ("diffusion-weight,w", po::value<float>(&diffusion_weight)->default_value(0), "Weight used to diffuse neighboring intensities into each other to slowly smooth the image. Disabled if zero, cannot exceed 0.25f.")
        ("contrast-on,c", po::value<float>(&contrast_on)->default_value(1.2f), "Contrast associated to ON events.")
        ("contrast-off", po::value<float>(&contrast_off)->default_value(-1), "Contrast associated to OFF events. If negative, the inverse of contrast-on is used.")
        ("tonemapping-count", po::value<int>(&tonemapping_max_ev_count)->default_value(5), "Maximum event count to tonemap in 8-bit grayscale frame. This needs to be adapted to the scene dynamic range & sensor sensitivity.")
        ("output-video,o", po::value<std::string>(&output_video_path)->default_value(""), "Save display window in a .avi format")
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

    if (contrast_off <= 0) {
        contrast_off = 1 / contrast_on;
    }

    // Open the input event stream
    Metavision::Camera camera;
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
    } else { // otherwise, set the input source to a camera
        try {
            if (!serial.empty()) {
                camera = Metavision::Camera::from_serial(serial);
            } else {
                camera = Metavision::Camera::from_first_available();
            }
            if (!cam_config_path.empty()) {
                camera.load(cam_config_path);
            }
        } catch (Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 3;
        }
    }

    const int camera_width  = camera.geometry().get_width();
    const int camera_height = camera.geometry().get_height();

    // Instantiate event integration algorithm
    bool show_cmap = false;
    Metavision::EventsIntegrationAlgorithm ev_integrator(camera_width, camera_height, decay_time, contrast_on,
                                                         contrast_off, tonemapping_max_ev_count,
                                                         integration_blur_radius, diffusion_weight);
    Metavision::ContrastMapGenerationAlgorithm cmap_generator(camera_width, camera_height, contrast_on, contrast_off);
    const float tonemapping_factor = 1 / static_cast<float>(std::pow(contrast_on, tonemapping_max_ev_count - 1) - 1);
    Metavision::OnDemandFrameGenerationAlgorithm frame_generator(camera_width, camera_height, decay_time / 3,
                                                                 Metavision::ColorPalette::Gray);

    // Instantiate window to render generated frames
    Metavision::Window window("Metavision Event Integration", camera_width, 2 * camera_height,
                              Metavision::BaseWindow::RenderMode::GRAY);
    window.set_keyboard_callback(
        [&window, &show_cmap](Metavision::UIKeyEvent key, int, Metavision::UIAction action, int) {
            if (action == Metavision::UIAction::RELEASE &&
                (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                window.set_close_flag();
            } else if (action == Metavision::UIAction::RELEASE && key == Metavision::UIKeyEvent::KEY_T) {
                show_cmap = !show_cmap;
            }
        });
    // Instantiate video writer
    std::unique_ptr<cv::VideoWriter> video_writer;
    if (!output_video_path.empty()) {
        const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        video_writer     = std::make_unique<cv::VideoWriter>(output_video_path, fourcc, 1e6 / period_us,
                                                             cv::Size(camera_width, 2 * camera_height), true);
        if (!video_writer->isOpened()) {
            MV_LOG_ERROR() << "Failed to open video writer!";
            return 1;
        }
    }

    // Instantiate event reslicer and define its slicing & event callbacks
    Metavision::EventBufferReslicerAlgorithm reslicer(
        nullptr, Metavision::EventBufferReslicerAlgorithm::Condition::make_n_us(period_us));

    cv::Mat_<float> contrast_map;
    cv::Mat visu_frame(2 * camera_height, camera_width, CV_8U), visu_frame_color;
    cv::Mat visu_events     = visu_frame(cv::Rect(0, 0, camera_width, camera_height));
    cv::Mat visu_integrated = visu_frame(cv::Rect(0, camera_height, camera_width, camera_height));
    reslicer.set_on_new_slice_callback([&](Metavision::EventBufferReslicerAlgorithm::ConditionStatus,
                                           Metavision::timestamp ts, std::size_t) {
        frame_generator.generate(ts, visu_events);
        if (show_cmap) {
            cmap_generator.generate(contrast_map);
            contrast_map.convertTo(visu_integrated, CV_8U, 128 * tonemapping_factor, 128 * (1 - tonemapping_factor));
        } else {
            ev_integrator.generate(visu_integrated);
        }

        window.show(visu_frame);
        if (video_writer) {
            cv::putText(visu_events, "CD Events", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 4);
            cv::putText(visu_integrated, "Integrated", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255),
                        4);
            cv::cvtColor(visu_frame, visu_frame_color, cv::COLOR_GRAY2BGR);
            video_writer->write(visu_frame_color);
        }
    });

    auto reslicer_ev_callback = [&](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
        frame_generator.process_events(ev_begin, ev_end);
        ev_integrator.process_events(ev_begin, ev_end);
        cmap_generator.process_events(ev_begin, ev_end);
    };

    // Register the processing callback when receiving new events
    camera.cd().add_callback([&](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
        reslicer.process_events(ev_begin, ev_end, reslicer_ev_callback);
    });

    // Start the event stream
    MV_LOG_INFO() << "Press 'ESC' to quit, 'T' to toggle between integrated events and contrast map display.";
    camera.start();
    while (camera.is_running() && !window.should_close()) {
        static constexpr std::int64_t kSleepPeriodMs = 20;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
    }
    camera.stop();
}
