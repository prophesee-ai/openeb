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

// This code sample demonstrates how to use Metavision Core SDK pipeline utility to display and filter events.
// It shows how to capture the keys pressed in a display window so as to modify the behavior of the stages while the
// pipeline is running.

#include <functional>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/core/pipeline/frame_generation_stage.h>
#include <metavision/sdk/core/algorithms/polarity_filter_algorithm.h>
#include <metavision/sdk/core/algorithms/roi_filter_algorithm.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/pipeline/frame_display_stage.h>

namespace po = boost::program_options;

/// [PIPELINE_FILTERING_BEGIN]
int main(int argc, char *argv[]) {
    std::string in_file_path;

    const std::string short_program_desc("Code sample showing how the pipeline utility can be used to "
                                         "create a simple application to filter and display events.\n");

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
        ("input-file,i", po::value<std::string>(&in_file_path), "Path to input file. If not specified, the camera live stream is used.")
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

    // A pipeline for which all added stages will automatically be run in their own processing threads (if applicable)
    Metavision::Pipeline p(true);

    // Construct a camera from a recording or a live stream
    Metavision::Camera cam;
    if (!in_file_path.empty()) {
        cam = Metavision::Camera::from_file(in_file_path);
    } else {
        cam = Metavision::Camera::from_first_available();
    }
    const unsigned short width  = cam.geometry().width();
    const unsigned short height = cam.geometry().height();

    /// Pipeline
    //
    //  0 (Camera) -->-- 1 (ROI) -->-- 2 (Polarity) -->-- 3 (Frame Generation) -->-- 4 (Display)
    //

    // 0) Stage producing events from a camera
    auto &cam_stage = p.add_stage(std::make_unique<Metavision::CameraStage>(std::move(cam)));

    // 1) Stage wrapping an ROI filter algorithm
    auto &roi_stage = p.add_algorithm_stage(
        std::make_unique<Metavision::RoiFilterAlgorithm>(80, 80, width - 80, height - 80, false), cam_stage, false);

    // 2) Stage wrapping a polarity filter algorithm
    auto &pol_stage = p.add_algorithm_stage(std::make_unique<Metavision::PolarityFilterAlgorithm>(0), roi_stage, false);

    // 3) Stage generating a frame from filtered events using accumulation time of 30ms
    auto &frame_stage = p.add_stage(std::make_unique<Metavision::FrameGenerationStage>(width, height, 30), pol_stage);

    // 4) Stage displaying the frame
    auto &disp_stage =
        p.add_stage(std::make_unique<Metavision::FrameDisplayStage>("CD events", width, height), frame_stage);

    disp_stage.set_key_callback([&](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
        if (action == Metavision::UIAction::RELEASE) {
            switch (key) {
            case Metavision::UIKeyEvent::KEY_A:
                // show all events
                pol_stage.set_enabled(false);
                break;
            case Metavision::UIKeyEvent::KEY_N:
                // show only negative events
                pol_stage.set_enabled(true);
                pol_stage.algo().set_polarity(0);
                break;
            case Metavision::UIKeyEvent::KEY_P:
                // show only positive events
                pol_stage.set_enabled(true);
                pol_stage.algo().set_polarity(1);
                break;
            case Metavision::UIKeyEvent::KEY_R:
                // toggle ROI filter
                roi_stage.set_enabled(!roi_stage.is_enabled());
                break;
            }
        }
    });

    // Run the pipeline
    p.run();

    return 0;
}
/// [PIPELINE_FILTERING_END]
