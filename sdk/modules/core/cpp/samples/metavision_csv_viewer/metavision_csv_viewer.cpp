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

// This code sample demonstrates how to use Metavision Core SDK pipeline utility to display events read from a CSV
// formatted event-based file, such as one produced by the sample metavision_file_to_csv. It also shows how to create a
// class deriving from BaseStage that produces data but consumes nothing.

#include <iostream>
#include <functional>
#include <fstream>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/core/pipeline/frame_generation_stage.h>
#include <metavision/sdk/core/algorithms/polarity_filter_algorithm.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/pipeline/frame_display_stage.h>

// A stage producing events from a CSV file with events written in the following format:
// x1,y1,t1,p1
// x2,y2,t2,p2
// ...
// xn,yn,tn,pn

/// [PIPELINE_USAGE_DEFINE_STAGE_BEGIN]
class CSVReadingStage : public Metavision::BaseStage {
public:
    CSVReadingStage(const std::string &filename) : ifs_(filename) {
        if (!ifs_.is_open()) {
            throw std::runtime_error("Unable to open " + filename);
        }

        cur_cd_buffer_ = cd_buffer_pool_.acquire();
        cur_cd_buffer_->clear();

        // this callback is called once the pipeline is started, so the stage knows it can start producing
        set_starting_callback([this]() {
            done_           = false;
            reading_thread_ = std::thread([this] { read(); });
        });

        // this callback is called once the pipeline is stopped : it can be initiated by a call
        // to @ref Pipeline::cancel() and/or after all stages are done and all task queues have been cleared
        set_stopping_callback([this]() {
            done_ = true;
            if (reading_thread_.joinable())
                reading_thread_.join();
        });
    }

    /// [PIPELINE_USAGE_READ_BEGIN]
    void read() {
        std::string line;
        while (!done_ && std::getline(ifs_, line)) {
            // once here, we know that the stage has not been stopped yet,
            // so we read a line and may produce a buffer
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (std::getline(iss, token, ',')) {
                tokens.push_back(token);
            }
            if (tokens.size() != 4) {
                MV_LOG_ERROR() << "Invalid line : <" << line << "> ignored";
            } else {
                if (cur_cd_buffer_->size() > 5000) {
                    // this is how a stage produces data to be consumed by the next stages
                    produce(cur_cd_buffer_);
                    cur_cd_buffer_ = cd_buffer_pool_.acquire();
                    cur_cd_buffer_->clear();
                }
                cur_cd_buffer_->emplace_back(static_cast<unsigned short>(std::stoul(tokens[0])),
                                             static_cast<unsigned short>(std::stoul(tokens[1])),
                                             static_cast<short>(std::stoi(tokens[2])), std::stoll(tokens[3]));
            }
        }
        // notifies to the pipeline that this producer has no more data to produce
        complete();
    }
    /// [PIPELINE_USAGE_READ_END]

private:
    std::atomic<bool> done_;
    std::thread reading_thread_;
    std::ifstream ifs_;
    EventBufferPool cd_buffer_pool_;
    EventBufferPtr cur_cd_buffer_;
};
/// [PIPELINE_USAGE_DEFINE_STAGE_END]

namespace po = boost::program_options;

/// [PIPELINE_USAGE_MAIN_BEGIN]
int main(int argc, char *argv[]) {
    std::string in_csv_file_path;
    int width, height;

    const std::string program_desc(
        "Code sample demonstrating how to use Metavision SDK to display events from a CSV file.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-csv-file,i", po::value<std::string>(&in_csv_file_path)->required(), "Path to input CSV file")
        ("width",            po::value<int>(&width)->default_value(640), "Width of the sensor associated to the CSV file")
        ("height",           po::value<int>(&height)->default_value(480), "Height of the sensor associated to the CSV file")
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

    /// Pipeline
    //
    //  0 (Csv Reading) -->-- 1 (Frame Generation) -->-- 2 (Display)
    //

    // A pipeline for which all added stages will automatically be run in their own processing threads (if applicable)
    Metavision::Pipeline p(true);

    // 0) Stage producing events from a CSV file
    auto &csv_stage = p.add_stage(std::make_unique<CSVReadingStage>(in_csv_file_path));

    // 1) Stage generating a frame with events previously produced using accumulation time of 10ms
    auto &frame_stage = p.add_stage(std::make_unique<Metavision::FrameGenerationStage>(width, height, 10), csv_stage);

    // 2) Stage displaying the generated frame
    auto &disp_stage =
        p.add_stage(std::make_unique<Metavision::FrameDisplayStage>("CD events", width, height), frame_stage);

    // Run the pipeline and wait for its completion
    p.run();

    return 0;
}
/// [PIPELINE_USAGE_MAIN_END]
