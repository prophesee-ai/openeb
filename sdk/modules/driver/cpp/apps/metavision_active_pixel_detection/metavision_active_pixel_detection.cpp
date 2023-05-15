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

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <signal.h>
#include <thread>
#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>
#if CV_MAJOR_VERSION >= 4
#include <opencv2/highgui/highgui_c.h>
#endif
#include <opencv2/imgproc.hpp>
#include <metavision/sdk/base/utils/generic_header.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/core/utils/colors.h>
#include <metavision/sdk/core/utils/misc.h>
#include <metavision/sdk/core/utils/rate_estimator.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_monitoring.h>
#include <metavision/hal/facilities/i_plugin_software_info.h>
#include <metavision/hal/utils/resources_folder.h>
#include <metavision/psee_hw_layer/devices/genx320/genx320_ll_roi.h>

static const int ESCAPE = 27;
static const int SPACE  = 32;

namespace po = boost::program_options;

namespace {
struct Data {
    cv::Mat counts;
    cv::Mat mask;
    double max, mean, stddev, threshold;
    int num_active_pixels;
    Metavision::Camera *camera = nullptr;

    Data(int width, int height) : counts(height, width, CV_32F), mask(height, width, CV_32F), threshold(10) {}
};

void onTrackbarCallback(int value, void *args) {
    Data *data = reinterpret_cast<Data *>(args);

    data->threshold = data->mean + value * 0.5 * data->stddev;
    data->counts.copyTo(data->mask);

    cv::threshold(data->counts, data->mask, data->threshold, 1.0, cv::THRESH_BINARY);
    cv::imshow("Active pixel detection", data->mask);

    data->num_active_pixels = cv::sum(data->mask)[0];
    double ratio            = static_cast<double>(data->num_active_pixels) / data->mask.total();
    MV_LOG_INFO() << Metavision::Log::no_space << std::fixed << std::setw(15) << std::setprecision(2) << std::right
                  << "Threshold :" << std::setw(15) << (data->threshold * data->max) << "  " << std::setw(15)
                  << "Active pixels: " << std::setw(15) << data->num_active_pixels << " (" << std::setw(7)
                  << (ratio * 100.0) << "% )";
}

void applyROI(const Data &data) {
    bool ret    = false;
    auto roi_ll = data.camera->get_device().get_facility<Metavision::GenX320LowLevelRoi>();
    if (roi_ll) {
        Metavision::GenX320LowLevelRoi::Grid roi_grid(data.mask.cols / 32, data.mask.cols);
        if (!data.mask.empty()) {
            for (int y = 0; y < data.mask.rows; ++y) {
                const float *ptr = data.mask.ptr<float>(y);
                for (int x = 0; x < data.mask.cols; ++x) {
                    if (ptr[x] > 0.5) {
                        roi_grid.set_pixel(x, y, false);
                    }
                }
            }
            roi_ll->apply(roi_grid);
            ret = true;
        }
    }
    if (ret) {
        MV_LOG_INFO() << "ROI based on the calibrated active pixel map successfully applied";
    } else {
        MV_LOG_ERROR() << "Unable to apply ROI based on the calibrated active pixel map";
    }
}

void saveDetectionData(const std::string &calib_output_path, const std::string &counts_output_path, const Data &data) {
    bool ret = false;
    std::ofstream ofs(calib_output_path);
    if (ofs.is_open()) {
        try {
            Metavision::GenericHeader header;
            header.add_date();
            const auto i_monitoring = data.camera->get_device().get_facility<Metavision::I_Monitoring>();
            if (i_monitoring) {
                try {
                    header.set_field("temperature", std::to_string(i_monitoring->get_temperature()));
                } catch (...) {}
                try {
                    header.set_field("illumination", std::to_string(i_monitoring->get_illumination()));
                } catch (...) {}
            }
            const auto i_ll_biases = data.camera->get_device().get_facility<Metavision::I_LL_Biases>();
            if (i_ll_biases) {
                std::map<std::string, int> biases = i_ll_biases->get_all_biases();
                for (auto it = biases.begin(), it_end = biases.end(); it != it_end; ++it) {
                    header.set_field(it->first, std::to_string(it->second));
                }
            }
            header.set_field("mean", std::to_string(data.mean * data.max));
            header.set_field("stddev", std::to_string(data.stddev * data.max));
            header.set_field("threshold", std::to_string(data.threshold * data.max));
            header.set_field("max", std::to_string(data.max));
            header.set_field("active_pixels_count", std::to_string(data.num_active_pixels));
            header.set_field("active_pixels_percentage",
                             std::to_string(data.num_active_pixels * 100. / data.counts.total()));

            ofs << header.to_string();
            ret = true;
        } catch (...) {}
    }
    if (ret) {
        for (int y = 0; y < data.mask.rows; ++y) {
            const float *ptr = data.mask.ptr<float>(y);
            for (int x = 0; x < data.mask.cols; ++x) {
                if (ptr[x] > 0.5) {
                    ofs << x << " " << y << "\n";
                }
            }
        }

        cv::Mat counts;
        data.counts.convertTo(counts, CV_8U, 255.0f);
        cv::imwrite(counts_output_path, counts);
        MV_LOG_INFO() << "Calibration results saved in" << calib_output_path;
        MV_LOG_INFO() << "Calibration data saved in" << counts_output_path;
    } else {
        ofs.close();
        std::remove(calib_output_path.c_str());
        MV_LOG_ERROR() << "Unable to save the calibration data";
    }
}

void putHelpMessages(cv::Mat &frame, const std::vector<std::string> &help_messages) {
    const auto aux_color = Metavision::get_bgr_color(Metavision::ColorPalette::Dark, Metavision::ColorType::Auxiliary);
    frame -= cv::Scalar::all(255 * 0.8);

    const int LetterWidth = 9, LineHeight = 20;
    cv::Size size;
    for (auto &msg : help_messages) {
        size.width = std::max<int>(size.width, msg.size() * LetterWidth);
        size.height += LineHeight;
    }

    int y_offset = 0;
    for (auto &msg : help_messages) {
        cv::putText(frame, msg, cv::Point((frame.cols - size.width) / 2, (frame.rows - size.height) / 2 + y_offset),
                    cv::FONT_HERSHEY_PLAIN, 1, aux_color, 1, cv::LINE_AA);
        y_offset += LineHeight;
    }
}

int processUI(int delay_ms) {
    auto then = std::chrono::high_resolution_clock::now();
    int key   = cv::waitKey(delay_ms);
    auto now  = std::chrono::high_resolution_clock::now();
    // cv::waitKey will not wait if no window is opened, so we wait for it, if needed
    std::this_thread::sleep_for(std::chrono::milliseconds(
        delay_ms - std::chrono::duration_cast<std::chrono::milliseconds>(now - then).count()));

    return key;
}

std::atomic<bool> signal_caught{false};

[[maybe_unused]] void signalHandler(int s) {
    MV_LOG_TRACE() << "Interrupt signal received." << std::endl;
    signal_caught = true;
}
} // anonymous namespace

int main(int argc, char *argv[]) {
    signal(SIGINT, signalHandler);

    const auto default_calib_output_path = Metavision::GenX320LowLevelRoi::default_calibration_path();
    std::string serial;
    std::string biases_file_path;
    std::string calib_output_path;
    const double default_duration = 0.1;
    double duration;
    const int default_min_event_count = 100000;
    int min_event_count;

    bool do_retry = false;

    const std::string short_program_desc("Application that detects and masks active pixels\n"
                                         "Warning: please note that this application is designed for future sensors "
                                         "and won't work with current ones (Gen31, Gen41 and IMX636)\n\n");
    const std::vector<std::string> help_messages = {"Press SPACE key to start the acquisition for the chosen duration",
                                                    "and active pixel detection.",
                                                    "Once the detection has been done, you can adjust the threshold",
                                                    "using the slider.",
                                                    "The active pixels are detected as the pixels for which a higher",
                                                    "number of events than the threshold have been received during",
                                                    "the acquisition. Hence, as the threshold is increased,",
                                                    "less pixels are considered active.",
                                                    "",
                                                    "You can then press the following keys :",
                                                    "Press 's' to save the calibration in the chosen path.",
                                                    "Press 'a' to apply a ROI masking the detected active pixels.",
                                                    "Press 'q' or Escape key to leave the program.",
                                                    "Press 'h' to show/hide this help message."};
    std::string help_desc;
    for (const auto &msg : help_messages) {
        if (!msg.empty()) {
            help_desc += std::string(msg) + "\n";
        }
    }
    const std::string long_program_desc = short_program_desc + help_desc;

    po::options_description options_desc("Options", 120, 80);
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("serial,s",          po::value<std::string>(&serial),"Serial ID of the camera. If not provided, the first available camera will be used.")
        ("biases,b",          po::value<std::string>(&biases_file_path), "Path to a biases file. If not specified, the camera will be configured with the default biases.")
        ("duration,d",        po::value<double>(&duration)->default_value(default_duration),"Duration in second of the calibration, the pixels will be analyzed during this period.")
        ("min-event-count,n", po::value<int>(&min_event_count)->default_value(default_min_event_count),"Minimum number of events to be acquired before starting the calibration.")
        ("output-calib,o",    po::value<std::string>(&calib_output_path)->default_value(default_calib_output_path.generic_string()), "Path to an output file used to save the result of the active pixels detection.\n"
                                                                                                                                          "The default path should be used if the calibration should be applied automatically when the camera starts.\n"
                                                                                                                                          "A custom path can be used to save the calibration data, but it won't be automatically loaded and applied when the camera starts.")
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

    if (calib_output_path != default_calib_output_path) {
        MV_LOG_WARNING() << "The specified path to save the calibration result is not the default one, the "
                            "calibration data will not be applied automatically when opening a camera.";
    }

    if (min_event_count <= 0) {
        MV_LOG_WARNING() << "Invalid acquisition min event count, setting it to" << default_min_event_count;
        min_event_count = default_min_event_count;
    }
    if (duration <= 0) {
        MV_LOG_WARNING() << "Invalid acquisition duration, setting it to" << default_duration;
        duration = default_duration;
    }

    std::string counts_output_path = std::regex_replace(calib_output_path, std::regex("\\.[^.]*$"), ".png");

    MV_LOG_INFO() << Metavision::Log::no_endline << long_program_desc;

    do {
        Metavision::Camera camera;
        bool camera_is_opened = false;
        bool show_help        = true;

        try {
            Metavision::DeviceConfig config;
            config.set("ignore_active_pixel_calibration_data", true);
            if (!serial.empty()) {
                camera = Metavision::Camera::from_serial(serial, config);
            } else {
                camera = Metavision::Camera::from_first_available(config);
            }

            if (biases_file_path != "") {
                camera.biases().set_from_file(biases_file_path);
            }

            camera_is_opened = true;
        } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }

        if (!camera_is_opened) {
            if (do_retry) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                MV_LOG_INFO() << "Trying to reopen camera...";
                continue;
            } else {
                return -1;
            }
        }

        // Add runtime error callback
        camera.add_runtime_error_callback([&do_retry](const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            do_retry = true;
        });

        // Get the geometry of the camera
        const auto &geometry = camera.geometry();

        // Setup CD event rate estimator
        double avg_rate;
        Metavision::RateEstimator cd_rate_estimator(
            [&avg_rate](Metavision::timestamp ts, double arate, double) { avg_rate = arate; }, 100000, 1000000, true);

        // Setup CD frame generator
        Metavision::CDFrameGenerator cd_frame_generator(geometry.width(), geometry.height());
        cd_frame_generator.set_display_accumulation_time_us(10000);

        std::mutex cd_frame_mutex;
        cv::Mat cd_frame;
        Metavision::timestamp cd_frame_ts{0};
        cd_frame_generator.start(
            30, [&cd_frame_mutex, &cd_frame, &cd_frame_ts](const Metavision::timestamp &ts, const cv::Mat &frame) {
                std::unique_lock<std::mutex> lock(cd_frame_mutex);
                cd_frame_ts = ts;
                cd_frame.create(std::max(frame.rows, 480), std::max(frame.cols, 640), CV_8UC3);
                cd_frame = Metavision::get_bgr_color(Metavision::ColorPalette::Dark, Metavision::ColorType::Background);
                cv::Mat sub_cd_frame = cd_frame(cv::Rect((cd_frame.cols - frame.cols) / 2,
                                                         (cd_frame.rows - frame.rows) / 2, frame.cols, frame.rows));
                frame.copyTo(sub_cd_frame);
            });

        // Setup CD frame display
        std::string cd_window_name("CD Events");
        cv::namedWindow(cd_window_name, cv::WINDOW_NORMAL);
        cv::setWindowProperty(cd_window_name, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
        cv::resizeWindow(cd_window_name, geometry.width(), geometry.height());
        cv::moveWindow(cd_window_name, 0, 0);
#if (CV_MAJOR_VERSION == 3 && (CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION) >= 408) || \
    (CV_MAJOR_VERSION == 4 && (CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION) >= 102)
        cv::setWindowProperty(cd_window_name, cv::WND_PROP_TOPMOST, 1);
#endif

        // Setup calibration frame display
        bool calib_setup = false;
        std::string calib_window_name("Active pixel detection");
        std::string calib_trackbar_name("Threshold");

        // Setup camera CD callback to update the frame generator and event rate estimator
        int cd_events_cb_id =
            camera.cd().add_callback([&cd_frame_generator, &cd_rate_estimator](const Metavision::EventCD *ev_begin,
                                                                               const Metavision::EventCD *ev_end) {
                cd_frame_generator.add_events(ev_begin, ev_end);
                cd_rate_estimator.add_data(std::prev(ev_end)->t, std::distance(ev_begin, ev_end));
            });

        // Start the camera streaming
        camera.start();

        std::vector<Metavision::EventCD> events;
        Data data(geometry.width(), geometry.height());
        int calib_cb_id = -1;
        while (!signal_caught && camera.is_running()) {
            {
                std::unique_lock<std::mutex> lock(cd_frame_mutex);
                if (!cd_frame.empty()) {
                    if (show_help) {
                        putHelpMessages(cd_frame, help_messages);
                    }
                    std::string text;
                    text = Metavision::getHumanReadableTime(cd_frame_ts);
                    text += "     ";
                    text += Metavision::getHumanReadableRate(avg_rate);
                    cv::putText(cd_frame, text, cv::Point(10, 20), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(108, 143, 255),
                                1, cv::LINE_AA);
                    cv::imshow(cd_window_name, cd_frame);
                }
            }
            if (events.size() > static_cast<size_t>(min_event_count) &&
                (events.front().t + duration * 1e6) < events.back().t) {
                MV_LOG_INFO() << "Acquisition done\n";
                camera.cd().remove_callback(calib_cb_id);

                if (!calib_setup) {
                    cv::namedWindow(calib_window_name, cv::WINDOW_NORMAL);
                    cv::setWindowProperty(calib_window_name, cv::WND_PROP_ASPECT_RATIO, cv::WINDOW_KEEPRATIO);
                    cv::resizeWindow(calib_window_name, geometry.width(), geometry.height());
                    cv::moveWindow(calib_window_name, 0, geometry.height());
#if (CV_MAJOR_VERSION == 3 && (CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION) >= 408) || \
    (CV_MAJOR_VERSION == 4 && (CV_MINOR_VERSION * 100 + CV_SUBMINOR_VERSION) >= 102)
                    cv::setWindowProperty(calib_window_name, cv::WND_PROP_TOPMOST, 1);
#endif
                    cv::createTrackbar(calib_trackbar_name, calib_window_name, nullptr, 10, onTrackbarCallback, &data);
                    // order is important here : max before min
                    cv::setTrackbarMax(calib_trackbar_name, calib_window_name, 10);
                    cv::setTrackbarMin(calib_trackbar_name, calib_window_name, 3);
                    calib_setup = true;
                }

                MV_LOG_INFO() << "Starting calibration with" << events.size() << "events...";
                double max = 0;
                data.counts.setTo(cv::Scalar::all(0));
                for (const auto &event : events) {
                    const double count = ++data.counts.at<float>(event.y, event.x);
                    max                = std::max(count, max);
                }
                data.max = max;
                data.counts /= data.max;
                events.clear();

                double mult = 3.0;
                cv::Scalar mean, stddev;
                cv::meanStdDev(data.counts, mean, stddev);
                data.mean      = mean[0];
                data.stddev    = stddev[0];
                data.threshold = data.mean + mult * data.stddev;
                data.camera    = &camera;
                MV_LOG_INFO() << "Calibration done";
                MV_LOG_INFO() << Metavision::Log::no_space << std::fixed << std::setw(15) << std::setprecision(2)
                              << std::right << "Max :" << std::setw(15) << static_cast<size_t>(data.max);
                MV_LOG_INFO() << Metavision::Log::no_space << std::fixed << std::setw(15) << std::setprecision(2)
                              << std::right << "Mean :" << std::setw(15) << (data.mean * data.max);
                MV_LOG_INFO() << Metavision::Log::no_space << std::fixed << std::setw(15) << std::setprecision(2)
                              << std::right << "Stddev :" << std::setw(15) << (data.stddev * data.max);

                int pos = std::round(mult / 0.5);
                cv::setTrackbarPos(calib_trackbar_name, calib_window_name, pos);
                onTrackbarCallback(pos, &data);
                applyROI(data);
                MV_LOG_INFO() << "Active pixels have been masked";
                saveDetectionData(calib_output_path, counts_output_path, data);
                MV_LOG_INFO() << "";

                MV_LOG_INFO()
                    << "You can manually adjust the threshold to see the effect on the active pixel detection map";
                MV_LOG_INFO() << "Press 'a' to apply a ROI based on selected threshold and mask out active pixels";
                MV_LOG_INFO() << "Press 's' to overwrite the saved calibration data with the selected threshold";
                MV_LOG_INFO() << Metavision::Log::no_space
                              << "\n-----------------------------------------------------------------------------------"
                                 "-----------------\n";
            }

            // Wait for a pressed key for 33ms, that means that the display is refreshed at 30 FPS
            int key = processUI(33);
            switch (key) {
            case 'q':
            case ESCAPE:
                camera.stop();
                do_retry = false;
                break;
            case SPACE: {
                auto log = MV_LOG_INFO() << Metavision::Log::no_space;
                log << "\n---------------------------------------------------------------------------------------------"
                       "-------\n";
                log << "Started acquisition for " << duration << "s...";
                calib_cb_id = camera.cd().add_callback(
                    [&events](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
                        events.insert(events.end(), ev_begin, ev_end);
                    });
                break;
            }
            case 'a':
                applyROI(data);
                break;
            case 's':
                saveDetectionData(calib_output_path, counts_output_path, data);
                break;
            case 'h':
                MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline << "\n" << help_desc;
                show_help = !show_help;
                break;
            default:
                break;
            }
        }

        // unregister callbacks to make sure they are not called anymore
        if (cd_events_cb_id >= 0) {
            camera.cd().remove_callback(cd_events_cb_id);
        }

        // Stop the camera streaming, optional, the destructor will automatically do it
        camera.stop();
        cd_frame_generator.stop();
    } while (!signal_caught && do_retry);

    return signal_caught;
}
