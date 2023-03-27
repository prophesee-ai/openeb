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

// This code sample demonstrates how to use Metavision Core SDK pipeline utility to filter events and show a frame
// combining unfiltered and filtered events. It also shows how to set a custom consuming callback on a
// @ref FrameCompositionStage instance, so that it can consume data from multiple stages.

#include <functional>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/algorithms/polarity_filter_algorithm.h>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/core/pipeline/frame_generation_stage.h>
#include <metavision/sdk/core/pipeline/frame_composition_stage.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/pipeline/frame_display_stage.h>

namespace po = boost::program_options;

/// [PIPELINE_COMPOSED_BEGIN]
int main(int argc, char *argv[]) {
    std::string in_file_path;

    const std::string program_desc("Code sample demonstrating how to use Metavision SDK CV to filter events\n"
                                   "and show a frame combining unfiltered and filtered events.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-file,i", po::value<std::string>(&in_file_path), "Path to input file. If not specified, the camera live stream is used.")
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

    // A pipeline for which all added stages will automatically be run in their own processing threads (if applicable)
    Metavision::Pipeline p(true);

    // Construct a camera from a file or a live stream
    Metavision::Camera cam;
    if (!in_file_path.empty()) {
        cam = Metavision::Camera::from_file(in_file_path);
    } else {
        cam = Metavision::Camera::from_first_available();
    }
    const unsigned short width  = cam.geometry().width();
    const unsigned short height = cam.geometry().height();

    const Metavision::timestamp event_buffer_duration_ms = 2;
    const uint32_t accumulation_time_ms                  = 10;
    const int display_fps                                = 100;

    /// Pipeline
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

    // 0) Stage producing events from a camera
    auto &cam_stage = p.add_stage(std::make_unique<Metavision::CameraStage>(std::move(cam), event_buffer_duration_ms));

    // 1) Stage wrapping a polarity filter algorithm to keep positive events
    auto &pol_filter_stage = p.add_algorithm_stage(std::make_unique<Metavision::PolarityFilterAlgorithm>(1), cam_stage);

    // 2,3) Stages generating frames from the previous stages
    auto &left_frame_stage =
        p.add_stage(std::make_unique<Metavision::FrameGenerationStage>(width, height, accumulation_time_ms), cam_stage);
    auto &right_frame_stage = p.add_stage(
        std::make_unique<Metavision::FrameGenerationStage>(width, height, accumulation_time_ms), pol_filter_stage);

    // 4) Stage generating a combined frame
    auto &full_frame_stage = p.add_stage(std::make_unique<Metavision::FrameCompositionStage>(display_fps));
    full_frame_stage.add_previous_frame_stage(left_frame_stage, 0, 0, width, height);
    full_frame_stage.add_previous_frame_stage(right_frame_stage, width + 10, 0, width, height);

    // 5) Stage displaying the combined frame
    const auto full_width  = full_frame_stage.frame_composer().get_total_width();
    const auto full_height = full_frame_stage.frame_composer().get_total_height();
    auto &disp_stage       = p.add_stage(
        std::make_unique<Metavision::FrameDisplayStage>("CD & noise filtered CD events", full_width, full_height),
        full_frame_stage);

    // Run the pipeline and wait for its completion
    p.run();

    return 0;
}
/// [PIPELINE_COMPOSED_END]
