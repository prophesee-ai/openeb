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

// Example of using Metavision SDK Driver and Core API for generating a video from a RAW file.

#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/cv_video_recorder.h>

namespace po = boost::program_options;

void remove_file(const std::string &filepath) {
    boost::filesystem::remove(boost::filesystem::path(filepath));
}

int main(int argc, char *argv[]) {
    std::string in_raw_file_path;
    std::string out_video_file_path;

    uint32_t accumulation_time;
    double slow_motion_factor;
    std::uint16_t fps;
    std::string fourcc;

    const std::string program_desc("Application to generate a video from RAW file.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i",     po::value<std::string>(&in_raw_file_path)->required(), "Path to input RAW file.")
        ("output-video-file,o",  po::value<std::string>(&out_video_file_path), "Path to output AVI file. If not provided, the base name of the input file will be used.")
        ("accumulation-time,a",  po::value<uint32_t>(&accumulation_time)->default_value(10000), "Accumulation time (in us).")
        ("fps",                  po::value<std::uint16_t>(&fps)->default_value(30), "Frame rate of the video to generate.")
        ("slow-motion-factor,s", po::value<double>(&slow_motion_factor)->default_value(1.), "Slow motion factor (or fast for value lower than 1) to apply to generate the video.")
        ("fourcc",               po::value<std::string>(&fourcc)->default_value("MJPG"), "Fourcc 4-character code of codec used to compress the frames. List of codes can be obtained at [Video Codecs by FOURCC](http://www.fourcc.org/codecs.php) page.")
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

    // Use the input filename as the output video name, if the output video name is empty
    if (out_video_file_path.empty()) {
        out_video_file_path = in_raw_file_path.substr(0, in_raw_file_path.find_last_of(".raw") - 3) + ".avi";
    }

    if (slow_motion_factor <= 0) {
        MV_LOG_ERROR() << "Input slow motion factor must be greater than 0. Got" << slow_motion_factor;
        return 1;
    }

    Metavision::Camera camera;

    try {
        camera = Metavision::Camera::from_file(in_raw_file_path, false);
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }

    // Get the geometry of the camera
    auto &geometry = camera.geometry();

    // Set up video write
    Metavision::CvVideoRecorder recorder(out_video_file_path,
                                         cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]), fps,
                                         cv::Size(geometry.width(), geometry.height()), true);

    recorder.start();

    // Set up frame generator
    Metavision::PeriodicFrameGenerationAlgorithm frame_generation(geometry.width(), geometry.height());
    frame_generation.set_accumulation_time_us(accumulation_time);
    frame_generation.set_fps(slow_motion_factor * fps);
    frame_generation.set_output_callback(
        [&](Metavision::timestamp frame_ts, cv::Mat &cd_frame) { recorder.write(cd_frame); });

    // Set up cd callback to process the events
    camera.cd().add_callback([&](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
        frame_generation.process_events(ev_begin, ev_end);
    });

    // Set up a change of status callback
    camera.add_status_change_callback([&recorder](const Metavision::CameraStatus &status) {
        // When the camera stops, we stop the recorder as well and wait for all the frames to be written.
        if (status == Metavision::CameraStatus::STOPPED) {
            recorder.stop();
        }
    });

    // Start the camera streaming
    camera.start();

    // Display a follow up message
    auto log = MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline;
    const std::string message("Generating video...");
    int dots = 0;

    // wait for the process to end
    while (recorder.is_recording()) {
        log << "\r" << message.substr(0, message.size() - 3 + dots) + std::string("   ").substr(0, 3 - dots)
            << std::flush;
        dots = (dots + 1) % 4;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    log << "\rVideo has been saved in " << out_video_file_path << std::endl;
    return 0;
}
