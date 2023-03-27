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

#include <cstring>
#include <iostream>
#include <list>
#include <thread>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <opencv2/imgcodecs.hpp>

#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/hal/facilities/i_event_frame_decoder.h>

#include "convert_event_frame.h"

namespace po = boost::program_options;

static float *process_event_frame_on_gpu(const Metavision::RawEventFrameHisto &histo) {
    auto cfg                = histo.get_config();
    const size_t histo_size = cfg.height * cfg.width;
    void *in_buff;

    cudaMallocManaged(&in_buff, histo_size);
    std::memcpy(in_buff, histo.get_data().data(), histo_size);

    float *out_buff;
    if (cfg.packed) {
        cudaMallocManaged(reinterpret_cast<void **>(&out_buff), histo_size * 3 * sizeof(float));
        convert_histogram(histo_size, reinterpret_cast<uint8_t *>(in_buff), out_buff,
                          cfg.channel_bit_size[Metavision::HistogramChannel::POSITIVE],
                          cfg.channel_bit_size[Metavision::HistogramChannel::NEGATIVE]);
    } else {
        cudaMallocManaged(reinterpret_cast<void **>(&out_buff), histo_size / 2 * 3 * sizeof(float));
        convert_histogram_padded(histo_size, reinterpret_cast<uint8_t *>(in_buff), out_buff,
                                 cfg.channel_bit_size[Metavision::HistogramChannel::POSITIVE],
                                 cfg.channel_bit_size[Metavision::HistogramChannel::NEGATIVE]);
    }

    cudaFree(in_buff);

    return out_buff;
}

static float *process_event_frame_on_gpu(const Metavision::RawEventFrameDiff &diff) {
    auto cfg               = diff.get_config();
    const size_t diff_size = cfg.height * cfg.width;
    void *in_buff;

    cudaMallocManaged(&in_buff, diff_size);
    std::memcpy(in_buff, diff.get_data().data(), diff_size);

    float *out_buff;
    cudaMallocManaged(reinterpret_cast<void **>(&out_buff), diff_size * 3 * sizeof(float));
    convert_diff(diff_size, reinterpret_cast<int8_t *>(in_buff), out_buff, cfg.bit_size);

    cudaFree(in_buff);

    return out_buff;
}

int main(int argc, char *argv[]) {
    std::string in_file_path;
    std::string output_dir;

    const std::string program_desc("Sample preprocessing Raw event frame on a GPU using CUDA");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i",     po::value<std::string>(&in_file_path)->required(), "Path to input file.")
        ("output-dir,o",     po::value<std::string>(&output_dir)->required(), "Output directory to store frames.");
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

    if (!boost::filesystem::exists(output_dir)) {
        if (!boost::filesystem::create_directories(output_dir)) {
            MV_LOG_ERROR() << "Failed to create output directory " << output_dir;
            return 1;
        }
    }
    if (!boost::filesystem::is_directory(output_dir)) {
        MV_LOG_ERROR() << "Output path '" << output_dir << "' is not a directory";
        return 1;
    }

    Metavision::Camera camera;

    try {
        camera = Metavision::Camera::from_file(in_file_path, Metavision::FileConfigHints().real_time_playback(false));
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }

    try {
        auto &histo_module = camera.frame_histo();
        histo_module.add_callback([&output_dir](const Metavision::RawEventFrameHisto &histo) {
            static int frame_idx = 0;
            auto res             = process_event_frame_on_gpu(histo);
            auto cfg             = histo.get_config();

            cv::Mat frame;
            frame = cv::Mat(cfg.height, cfg.width, CV_32FC3, res);
            cv::imwrite(
                (boost::filesystem::path(output_dir) / ("histo_frame" + std::to_string(frame_idx) + ".jpg")).string(),
                frame);

            cudaFree(res);

            ++frame_idx;
        });
    } catch (...) {}

    try {
        auto &diff_module = camera.frame_diff();
        diff_module.add_callback([&output_dir](const Metavision::RawEventFrameDiff &diff) {
            static int frame_idx = 0;
            auto res             = process_event_frame_on_gpu(diff);
            auto cfg             = diff.get_config();

            cv::Mat frame;
            frame = cv::Mat(cfg.height, cfg.width, CV_32FC3, res);

            cv::imwrite(
                (boost::filesystem::path(output_dir) / ("diff_frame" + std::to_string(frame_idx) + ".jpg")).string(),
                frame);
            ++frame_idx;
        });
    } catch (...) {}

    camera.start();
    while (camera.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    camera.stop();

    return 0;
}
