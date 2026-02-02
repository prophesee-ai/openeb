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

// This code sample demonstrates how to generate and display histo / diff event frames.
// It also implements the capability to convert an event-based record to a histo / diff RAW file.

#include <fstream>
#include <functional>
#include <thread>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/algorithms/event_buffer_reslicer_algorithm.h>
#include <metavision/sdk/core/algorithms/event_frame_diff_generation_algorithm.h>
#include <metavision/sdk/core/algorithms/event_frame_histo_generation_algorithm.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/sdk/stream/camera.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    //////////////////////////////////////////////
    // Parse command line options
    //////////////////////////////////////////////
    std::string event_file_path, out_file_path, out_video_path;
    bool enable_histo, enable_diff, diff_allow_rollover, histo_packed, disable_display;
    unsigned int histo_bit_size_neg, histo_bit_size_pos, diff_bit_size;
    int nevents;
    Metavision::timestamp period_us, min_generation_period_us;

    const std::string short_program_desc("Code sample showing how to generate histo and diff event frames from an "
                                         "event stream and how to visualize them.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i", po::value<std::string>(&event_file_path)->required(), "Path to input file.")
        ("output-file,o", po::value<std::string>(&out_file_path)->default_value(""), "If specified, path to the event frame RAW file to be generated from the input event stream.")
        ("output-video,v", po::value<std::string>(&out_video_path)->default_value(""), "If specified, path to the video to be generated from the visualization.")
        ("period,p", po::value<Metavision::timestamp>(&period_us)->default_value(10000), "Period for the generation of the event frames, in us. If negative, only event numbers will be used.")
        ("nevents,n", po::value<int>(&nevents)->default_value(-1), "Number of events for the generation of the event frames. If negative, only time period will be used.")
        ("min-generation-period", po::value<Metavision::timestamp>(&min_generation_period_us)->default_value(0), "Minimum duration between the generation of 2 successive event frames, in us. This is 0 by default, meaning there is no minimum.")
        ("histo", po::bool_switch(&enable_histo), "If specified, enables the generation of histo event frames.")
        ("histo-bit-size-neg", po::value<unsigned int>(&histo_bit_size_neg)->default_value(4), "In histo mode, number of bits used to count OFF events. This must be strictly positive and the sum of bit sizes for OFF & ON events must be lower than 8.")
        ("histo-bit-size-pos", po::value<unsigned int>(&histo_bit_size_pos)->default_value(4), "In histo mode, number of bits used to count ON events. This must be strictly positive and the sum of bit sizes for OFF & ON events must be lower than 8.")
        ("histo-packed", po::bool_switch(&histo_packed), "In histo mode, enables packed OFF & ON channels.")
        ("diff", po::bool_switch(&enable_diff), "If specified, enables the generation of diff event frames.")
        ("diff-bit-size", po::value<unsigned int>(&diff_bit_size)->default_value(8), "In diff mode, number of bits used to sum event polarities. This must be greater than 2 and lower than 8.")
        ("diff-allow-rollover", po::bool_switch(&diff_allow_rollover), "In diff mode, flag to disable saturation and enable overflow / underflow of the counter.")
        ("no-display", po::bool_switch(&disable_display), "Disables visual feedback.")
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

    if (period_us <= 0 && nevents <= 0) {
        MV_LOG_ERROR() << "One of the event frame generation period or event number must be strictly positive.";
        return 1;
    }

    if (!enable_histo && !enable_diff) {
        MV_LOG_ERROR() << "One of Histo or Diff modes must be activated.";
        return 1;
    }

    if (enable_diff && enable_histo && out_file_path != "") {
        MV_LOG_ERROR()
            << "Error: output file generation is not supported when both diff and histo generation are enabled.";
        return 1;
    }

    if (enable_histo &&
        (histo_bit_size_neg <= 0 || histo_bit_size_pos <= 0 || histo_bit_size_neg + histo_bit_size_pos > 8)) {
        MV_LOG_ERROR()
            << "Error: in histo mode, the bit sizes must be strictly positive and their sum must be lower than 8.";
        return 1;
    }

    if (enable_diff && (diff_bit_size < 2 || diff_bit_size > 8)) {
        MV_LOG_ERROR() << "Error: in diff mode, the bit size must be in the range [2;8].";
        return 1;
    }

    //////////////////////////////////////////////
    // Instantiate processing pipeline
    //////////////////////////////////////////////

    // Instantiate Camera object from provided recording
    Metavision::Camera camera;
    try {
        camera = Metavision::Camera::from_file(event_file_path, Metavision::FileConfigHints().real_time_playback(true));

    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 2;
    }

    const int camera_width  = camera.geometry().get_width();
    const int camera_height = camera.geometry().get_height();
    const int npixels       = camera_width * camera_height;

    // Instantiate generator for event frame diff and visualization functor
    Metavision::EventFrameDiffGenerationAlgorithm<const Metavision::EventCD *> diff_generator(
        camera_width, camera_height, diff_bit_size, diff_allow_rollover, min_generation_period_us);
    Metavision::RawEventFrameDiff evframe_diff;
    const double diff_rescaler = 128. / (1 << (diff_bit_size - 1));
    auto diff_to_viz_fct       = [&](int8_t diff_val) {
        const uchar viz_val = cv::saturate_cast<uchar>(128 + diff_val * diff_rescaler);
        return cv::Vec3b(viz_val, viz_val, viz_val);
    };

    // Instantiate generator for event frame histo and visualization functor
    Metavision::EventFrameHistoGenerationAlgorithm<const Metavision::EventCD *> histo_generator(
        camera_width, camera_height, histo_bit_size_neg, histo_bit_size_pos, histo_packed, min_generation_period_us);
    Metavision::RawEventFrameHisto evframe_histo;
    const uint8_t histo_packed_neg_mask  = (1 << histo_bit_size_neg) - 1;
    const uint8_t histo_packed_pos_mask  = (1 << histo_bit_size_pos) - 1;
    const double histo_rescaler_neg      = 255. / histo_packed_neg_mask;
    const double histo_rescaler_pos      = 255. / histo_packed_pos_mask;
    auto histo_unpacked_chans_to_viz_fct = [&](uint8_t histo_val_neg, uint8_t histo_val_pos) {
        const uchar viz_val_neg = cv::saturate_cast<uchar>(histo_val_neg * histo_rescaler_neg);
        const uchar viz_val_pos = cv::saturate_cast<uchar>(histo_val_pos * histo_rescaler_pos);
        return cv::Vec3b(viz_val_neg, 0, viz_val_pos);
    };
    auto histo_packed_chans_to_viz_fct = [&](uint8_t histo_packed_val) {
        const uchar viz_val_neg =
            cv::saturate_cast<uchar>((histo_packed_val & histo_packed_neg_mask) * histo_rescaler_neg);
        const uchar viz_val_pos = cv::saturate_cast<uchar>(
            ((histo_packed_val >> histo_bit_size_neg) & histo_packed_pos_mask) * histo_rescaler_pos);
        return cv::Vec3b(viz_val_neg, 0, viz_val_pos);
    };

    // Instantiate visualization frames, display window and video writer
    const int visu_width  = (enable_diff ? camera_width : 0) + (enable_histo ? camera_width : 0);
    const int visu_height = camera_height;

    cv::Mat visu_diff, visu_histo, combined_frame(visu_height, visu_width, CV_8UC3);
    int x_offset = 0;
    if (enable_diff) {
        visu_diff = combined_frame(cv::Rect(0, 0, camera_width, camera_height));
        x_offset += camera_width;
    }
    if (enable_histo)
        visu_histo = combined_frame(cv::Rect(x_offset, 0, camera_width, camera_height));

    std::unique_ptr<Metavision::Window> window;
    if (!disable_display) {
        window = std::make_unique<Metavision::Window>("Metavision Event Frame Generation", visu_width, visu_height,
                                                      Metavision::BaseWindow::RenderMode::BGR);
        window->set_keyboard_callback(
            [&window](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE &&
                    (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                    window->set_close_flag();
                }
            });
    }

    std::unique_ptr<cv::VideoWriter> video_writer;
    if (out_video_path != "") {
        video_writer = std::make_unique<cv::VideoWriter>(out_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                                         30, cv::Size(visu_width, visu_height));
    }

    // Prepare output file
    std::unique_ptr<std::ofstream> file_writer;
    if (out_file_path != "") {
        file_writer = std::make_unique<std::ofstream>(out_file_path, std::ios::binary | std::ios::trunc);
        if (!file_writer->is_open()) {
            MV_LOG_ERROR() << "Error: failed to open specified output file '" << out_file_path << "'!";
            return 1;
        }
        auto header = camera.get_device().get_facility<Metavision::I_HW_Identification>()->get_header();
        header.add_date();
        header.remove_field("evt");
        std::ostringstream oss;
        if (enable_diff) {
            header.set_field("format", "DIFF3D");
            oss << "0p/" << diff_bit_size << "n";
            header.set_field("pixellayout", oss.str());
        } else {
            header.set_field("format", "HISTO3D");
            oss << histo_bit_size_pos << "p/" << histo_bit_size_neg << "n";
            header.set_field("pixellayout", oss.str());
            header.set_field("pixelbytes", histo_packed ? "1" : "2");
        }
        *file_writer << header;
    }

    // Instantiate event reslicer and define its slicing & event callbacks
    Metavision::EventBufferReslicerAlgorithm::Condition condition;
    if (period_us > 0 && nevents > 0)
        condition = Metavision::EventBufferReslicerAlgorithm::Condition::make_mixed(period_us, nevents);
    else if (period_us > 0)
        condition = Metavision::EventBufferReslicerAlgorithm::Condition::make_n_us(period_us);
    else
        condition = Metavision::EventBufferReslicerAlgorithm::Condition::make_n_events(nevents);
    Metavision::EventBufferReslicerAlgorithm reslicer(nullptr, condition);

    reslicer.set_on_new_slice_callback([&](Metavision::EventBufferReslicerAlgorithm::ConditionStatus status,
                                           Metavision::timestamp ts, std::size_t nevents) {
        std::ostringstream oss;
        // Visualize event frame diff
        if (enable_diff) {
            const bool diff_generated = diff_generator.generate(ts, evframe_diff);
            if (window || video_writer) {
                if (diff_generated) {
                    for (int y = 0; y < camera_height; ++y) {
                        auto it_diff_line = evframe_diff.get_data().cbegin() + y * camera_width;
                        std::transform(it_diff_line, it_diff_line + camera_width, visu_diff.ptr<cv::Vec3b>(y),
                                       diff_to_viz_fct);
                    }
                } else {
                    visu_diff.setTo(cv::Scalar(192, 192, 192));
                }
                oss.str("");
                oss << "diff " << diff_bit_size << (diff_allow_rollover ? " rollover" : "");
                cv::putText(visu_diff, oss.str(), cv::Point(10, camera_height - 20), cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(0, 0, 0));
            }
            if (file_writer && diff_generated) {
                file_writer->write(reinterpret_cast<char *>(evframe_diff.get_data().data()),
                                   evframe_diff.get_data().size());
            }
        }
        // Visualize event frame histo
        if (enable_histo) {
            const bool histo_generated = histo_generator.generate(ts, evframe_histo);
            if (window || video_writer) {
                if (histo_generated) {
                    if (histo_packed) {
                        for (int y = 0; y < camera_height; ++y) {
                            auto it_histo_line = evframe_histo.get_data().cbegin() + y * camera_width;
                            std::transform(it_histo_line, it_histo_line + camera_width, visu_histo.ptr<cv::Vec3b>(y),
                                           histo_packed_chans_to_viz_fct);
                        }
                    } else {
                        for (int y = 0; y < camera_height; ++y) {
                            auto it_histo_line_neg = evframe_histo.get_data().cbegin() + y * camera_width * 2;
                            auto it_histo_line_pos = evframe_histo.get_data().cbegin() + y * camera_width * 2 + 1;
                            for (int i = 0; i < camera_width; ++i) {
                                visu_histo.ptr<cv::Vec3b>(y)[i] =
                                    histo_unpacked_chans_to_viz_fct(it_histo_line_neg[2 * i], it_histo_line_pos[2 * i]);
                            }
                        }
                    }
                } else {
                    visu_histo.setTo(cv::Scalar(64, 64, 64));
                }
                oss.str("");
                oss << "histo n" << histo_bit_size_neg << "p" << histo_bit_size_pos;
                cv::putText(visu_histo, oss.str(), cv::Point(10, camera_height - 20), cv::FONT_HERSHEY_SIMPLEX, 1,
                            cv::Scalar(255, 255, 255));
            }
            if (file_writer && histo_generated) {
                file_writer->write(reinterpret_cast<char *>(evframe_histo.get_data().data()),
                                   evframe_histo.get_data().size());
            }
        }
        // Add event frame timestamp and display + append to video
        if (window || video_writer) {
            oss.str("");
            oss << ts;
            cv::putText(combined_frame, oss.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                        enable_diff ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255));
        }
        if (window)
            window->show(combined_frame);
        if (video_writer)
            video_writer->write(combined_frame);
    });

    auto aggregate_events_fct = [&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        if (enable_diff)
            diff_generator.process_events(begin, end);
        if (enable_histo)
            histo_generator.process_events(begin, end);
    };

    // Set the event processing callback
    camera.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        reslicer.process_events(begin, end, aggregate_events_fct);
    });

    //////////////////////////////////////////////
    // Run pipeline
    //////////////////////////////////////////////
    camera.start();
    if (!window) {
        while (camera.is_running()) {
            std::this_thread::yield();
        }
    } else {
        while (camera.is_running() && !window->should_close()) {
            Metavision::EventLoop::poll_and_dispatch(20);
        }
    }
    camera.stop();
    if (video_writer) {
        video_writer->release();
        MV_LOG_INFO() << "Wrote video file to:" << out_video_path;
    }
    if (file_writer) {
        file_writer->close();
        MV_LOG_INFO() << "Wrote event frame file to:" << out_file_path;
    }
}
