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

// Example of using Metavision SDK Driver and Core API for generating a video from a RAW or HDF5 file.

#include <iostream>
#include <regex>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/facilities/i_event_frame_decoder.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/cv_video_recorder.h>
#include <metavision/sdk/core/utils/raw_event_frame_converter.h>

namespace po = boost::program_options;

void remove_file(const std::string &filepath) {
    boost::filesystem::remove(boost::filesystem::path(filepath));
}

int main(int argc, char *argv[]) {
    std::string in_file_path;
    std::string out_video_file_path;

    uint32_t accumulation_time;
    double slow_motion_factor;
    const std::uint16_t fps(30);
    std::string fourcc;

    const std::string program_desc(
        "Application to generate a video from a RAW or HDF5 file.\n\n"
        "The frame rate of the output video (display rate) is fixed to 30.\n"
        "The frame rate to generate the frames from the events (generation rate) is driven by the slow motion factor "
        "(-s option): slow motion factor = generation rate / display rate\n"
        "Hence, generation rate = slow motion factor x 30.\n"
        "For example, to create a video from frames generated at 1500 FPS that will be rendered in slow-motion at 30 "
        "FPS, one has to set a slow-motion factor of 50.\n"
        "Note that in that case, the accumulation time (-a option) should be adapted accordingly. "
        "For example to 666us (1/1500) if you want each event to appear in a single frame, \n"
        "or 6666us (10/1500) if you want each event to appear in 10 frames");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i",     po::value<std::string>(&in_file_path)->required(), "Path to input file.")
        ("output-video-file,o",  po::value<std::string>(&out_video_file_path), "Path to output AVI file. If not provided, the base name of the input file will be used. The output video fps is fixed to 30.")
        ("accumulation-time,a",  po::value<uint32_t>(&accumulation_time)->default_value(10000), "Accumulation time (in us).")
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
        out_video_file_path = std::regex_replace(in_file_path, std::regex("\\.[^.]*$"), ".avi");
    }

    if (slow_motion_factor <= 0) {
        MV_LOG_ERROR() << "Input slow motion factor must be greater than 0. Got" << slow_motion_factor;
        return 1;
    }

    Metavision::Camera camera;

    try {
        camera = Metavision::Camera::from_file(in_file_path, Metavision::FileConfigHints().real_time_playback(false));
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }

    // Get the geometry of the camera
    auto &geometry = camera.geometry();

    // Set up video write
    cv::redirectError([](auto...) { return 0; }); // disable default OpenCV error display, we handle it ourselves
    const bool enable_image_sequence = (out_video_file_path.find('%') != std::string::npos);
    Metavision::CvVideoRecorder recorder(
        out_video_file_path,
        enable_image_sequence ? 0 : cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]),
        enable_image_sequence ? 0 : fps, cv::Size(geometry.width(), geometry.height()), true);

    recorder.start();

    // Set up frame generator
    Metavision::PeriodicFrameGenerationAlgorithm frame_generation(geometry.width(), geometry.height());
    bool has_cd = false;
    try {
        auto &cd = camera.cd();

        has_cd = true;
        frame_generation.set_accumulation_time_us(accumulation_time);
        frame_generation.set_fps(slow_motion_factor * fps);
        frame_generation.set_output_callback(
            [&recorder](Metavision::timestamp frame_ts, cv::Mat &cd_frame) { recorder.write(cd_frame); });

        // Set up cd callback to process the events
        cd.add_callback([&](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
            frame_generation.process_events(ev_begin, ev_end);
        });
    } catch (...) {}

    try {
        auto &histo_module = camera.frame_histo();
        Metavision::RawEventFrameConverter frame_converter(geometry.height(), geometry.width(), 2);
        histo_module.add_callback([&recorder, frame_converter](const Metavision::RawEventFrameHisto &histo) {
            auto histo_cfg       = histo.get_config();
            auto converted_histo = frame_converter.convert<float>(histo);
            cv::Mat histo_frame  = cv::Mat(histo_cfg.height, histo_cfg.width, CV_8UC3);

            for (unsigned row = 0; row < histo_cfg.height; ++row) {
                for (unsigned col = 0; col < histo_cfg.width; ++col) {
                    uint8_t val = ((*converted_histo)(col, row, Metavision::HistogramChannel::POSITIVE) -
                                   (*converted_histo)(col, row, Metavision::HistogramChannel::NEGATIVE)) *
                                      16 +
                                  127;
                    histo_frame.at<cv::Vec3b>(row, col) = {val, val, val};
                }
            }
            recorder.write(histo_frame);
        });
    } catch (...) {}

    try {
        auto &diff_module = camera.frame_diff();
        Metavision::RawEventFrameConverter frame_converter(geometry.height(), geometry.width(), 1);
        diff_module.add_callback([&recorder, frame_converter](const Metavision::RawEventFrameDiff &diff) {
            auto diff_cfg       = diff.get_config();
            auto converted_diff = frame_converter.convert<float>(diff);
            cv::Mat diff_frame  = cv::Mat(diff_cfg.height, diff_cfg.width, CV_8UC3);

            for (unsigned row = 0; row < diff_cfg.height; ++row) {
                for (unsigned col = 0; col < diff_cfg.width; ++col) {
                    uint8_t val                        = (*converted_diff)(col, row) * 16 + 127;
                    diff_frame.at<cv::Vec3b>(row, col) = {val, val, val};
                }
            }
            recorder.write(diff_frame);
        });
    } catch (...) {}

    // Set up a change of status callback
    camera.add_status_change_callback([&recorder, &frame_generation, has_cd](const Metavision::CameraStatus &status) {
        // When the camera stops, we stop the recorder as well and wait for all the frames to be written.
        if (status == Metavision::CameraStatus::STOPPED) {
            if (has_cd) {
                frame_generation.force_generate();
            }
        }
    });

    // Start the camera streaming
    camera.start();

    // Display a follow up message
    auto log = MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline;
    const std::string message("Generating video...");
    int dots = 0;

    // wait for the camera to finish streaming all date
    while (camera.is_running()) {
        log << "\r" << message.substr(0, message.size() - 3 + dots) + std::string("   ").substr(0, 3 - dots)
            << std::flush;
        dots = (dots + 1) % 4;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Stop the video recording process, this will save all the remaining frames (if any) and close the file
    try {
        recorder.stop();
    } catch (cv::Exception &e) {
        log << "\r" << std::endl;
        switch (e.code) {
        case cv::Error::StsOutOfRange: {
            if (std::string(e.what()).find("chunk size is out of bounds") != std::string::npos) {
                MV_LOG_ERROR()
                    << "There was an error while saving the video : output file is too large, try converting the "
                       "sequence with a smaller slow motion factor.";
                return 1;
            }
            break;
        }
        default:
            break;
        }
        MV_LOG_ERROR() << "There was an unexpected error while saving the video";
        return 1;
    }

    log << "\rThe video has been saved in " << out_video_file_path << std::endl;
    return 0;
}
