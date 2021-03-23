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

// This code sample demonstrates how to use Metavision SDK Driver module to show a mosaic of frames combining
// events and processed events from multiple raw streams.

#include <iostream>
#include <functional>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/core/pipeline/frame_generation_stage.h>
#include <metavision/sdk/core/pipeline/frame_composition_stage.h>
#include <metavision/sdk/core/algorithms/polarity_filter_algorithm.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/pipeline/frame_display_stage.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::vector<std::string> in_raw_file_paths;

    const std::string program_desc(
        "Code sample demonstrating how to use Metavision SDK Driver to show a mosaic of frames combining events \n"
        "from multiple raw streams and the results of their processing.\n"
        "Here, on the left, we show events from the input files and on the right, the result of polarity filter.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
 		("input-raw-file,i", po::value<std::vector<std::string>>(&in_raw_file_paths)->multitoken()->required(), "Paths to input RAW files.")
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

    // Pipeline
    //
    //  00 (Cam) ------>----- 10 (Pol)       01 (Cam) ------>----- 11 (Pol)             0n (Cam) ------>----- 1n (Pol)
    //  |                     |              |                     |                    |                     |
    //  v                     v              v                     v                    v                     v
    //  |                     |              |                     |                    |                     |
    //  20 (Frame Generators) 30             21 (Frame Generators) 31          ...      2n (Frame Generators) 3n
    //  |                     |              |                     |                    |                     |
    //  v                     v              v                     v                    v                     v
    //  |                     |              |                     |                    |                     |
    //  |-------------->------------------------------------>------------------>-<---------------------<------|
    //                                                                          |
    //                                                                          4 (Frame Composer)
    //                                                                          |
    //                                                                          v
    //                                                                          |
    //                                                                          5 (Display)
    //

    // 4) Frame composer stage
    auto &full_frame_stage = p.add_stage(std::make_unique<Metavision::FrameCompositionStage>(30));

    int w = 320, h = 240;
    const size_t count = in_raw_file_paths.size();
    for (size_t i = 0; i < count; ++i) {
        // Construct a camera from a file
        Metavision::Camera cam      = Metavision::Camera::from_file(in_raw_file_paths[i]);
        const unsigned short width  = cam.geometry().width();
        const unsigned short height = cam.geometry().height();

        // 0i) Stage producing events from a camera
        auto &cam_stage = p.add_stage(std::make_unique<Metavision::CameraStage>(std::move(cam)));

        // 1i) Stage wrapping a polarity filter algorithm
        auto &pol_stage = p.add_algorithm_stage(std::make_unique<Metavision::PolarityFilterAlgorithm>(0), cam_stage);

        // 2i,3i) Stages generating frames from the previous stages
        auto &left_frame_stage =
            p.add_stage(std::make_unique<Metavision::FrameGenerationStage>(width, height, 30), cam_stage);
        auto &right_frame_stage =
            p.add_stage(std::make_unique<Metavision::FrameGenerationStage>(width, height, 30), pol_stage);

        full_frame_stage.add_previous_frame_stage(left_frame_stage, 0, static_cast<int>(i * (h + 10)), w, h);
        full_frame_stage.add_previous_frame_stage(right_frame_stage, w + 10, static_cast<int>(i * (h + 10)), w, h);
    }

    // 5) Stage displaying the combined frame
    const int full_width  = full_frame_stage.frame_composer().get_total_width();
    const int full_height = full_frame_stage.frame_composer().get_total_height();
    auto &disp_stage =
        p.add_stage(std::make_unique<Metavision::FrameDisplayStage>("CD & negative CD events", full_width, full_height),
                    full_frame_stage);

    p.run();

    return 0;
}
